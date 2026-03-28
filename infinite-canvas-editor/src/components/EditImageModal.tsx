import { useState, useEffect } from 'react'
import { X } from 'lucide-react'

interface EditImageModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: (count: number, prompts: string[]) => void
  defaultCount?: number
  defaultPrompt?: string
  isDark?: boolean
}

export function EditImageModal({
  isOpen,
  onClose,
  onConfirm,
  defaultCount = 3,
  defaultPrompt = '',
  isDark = true,
}: EditImageModalProps) {
  const [count, setCount] = useState(defaultCount)
  const [prompts, setPrompts] = useState<string[]>(Array(defaultCount).fill(defaultPrompt))

  // 当数量变化时，调整 prompts 数组
  useEffect(() => {
    setPrompts((prev) => {
      const newPrompts = [...prev]
      if (count > prev.length) {
        // 增加输入框，用最后一个值填充
        const lastValue = prev[prev.length - 1] || defaultPrompt
        while (newPrompts.length < count) {
          newPrompts.push(lastValue)
        }
      } else if (count < prev.length) {
        // 减少输入框
        return newPrompts.slice(0, count)
      }
      return newPrompts
    })
  }, [count, defaultPrompt])

  if (!isOpen) return null

  const handleConfirm = () => {
    const validCount = Math.max(1, Math.min(10, count))
    onConfirm(validCount, prompts.map(p => p.trim()))
    onClose()
  }

  const updatePrompt = (index: number, value: string) => {
    setPrompts(prev => {
      const newPrompts = [...prev]
      newPrompts[index] = value
      return newPrompts
    })
  }

  const overlay = 'fixed inset-0 bg-black/60 flex items-center justify-center z-50'
  const modal = `w-[480px] max-h-[80vh] overflow-y-auto rounded-lg shadow-xl p-4 ${isDark ? 'bg-zinc-900 text-zinc-100' : 'bg-white text-zinc-900'}`
  const input = `w-full px-3 py-2 rounded border text-sm ${
    isDark ? 'bg-zinc-800 border-zinc-700 text-zinc-100' : 'bg-white border-zinc-300'
  }`
  const label = `block text-xs mb-1 ${isDark ? 'text-zinc-400' : 'text-zinc-600'}`
  const btnPrimary = `px-4 py-2 rounded text-sm font-medium ${
    isDark ? 'bg-violet-600 text-white hover:bg-violet-500' : 'bg-violet-600 text-white hover:bg-violet-700'
  }`
  const btnSecondary = `px-4 py-2 rounded text-sm font-medium ${
    isDark ? 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600' : 'bg-zinc-200 text-zinc-700 hover:bg-zinc-300'
  }`

  return (
    <div className={overlay} onClick={onClose}>
      <div className={modal} onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-semibold">Edit Image Settings</h3>
          <button
            onClick={onClose}
            className={`p-1 rounded ${isDark ? 'hover:bg-zinc-800' : 'hover:bg-zinc-100'}`}
          >
            <X className="size-4" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className={label}>Number of images (1-10)</label>
            <input
              type="number"
              min={1}
              max={10}
              value={count}
              onChange={(e) => setCount(parseInt(e.target.value) || 1)}
              className={input}
            />
          </div>

          <div className="space-y-3">
            <label className={label}>Prompts for each image</label>
            {prompts.map((prompt, index) => (
              <div key={index} className="flex gap-2">
                <span className={`text-xs pt-2 w-6 ${isDark ? 'text-zinc-500' : 'text-zinc-400'}`}>
                  {index + 1}.
                </span>
                <textarea
                  value={prompt}
                  onChange={(e) => updatePrompt(index, e.target.value)}
                  placeholder={`Prompt for image ${index + 1}...`}
                  rows={2}
                  className={`${input} resize-none flex-1`}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="flex justify-end gap-2 mt-6">
          <button onClick={onClose} className={btnSecondary}>
            Cancel
          </button>
          <button onClick={handleConfirm} className={btnPrimary}>
            Confirm
          </button>
        </div>
      </div>
    </div>
  )
}
