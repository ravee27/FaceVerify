import { useRef } from 'react'

export default function UploadBox({ label, onFile }: { label: string, onFile: (f: File) => void }) {
  const ref = useRef<HTMLInputElement>(null)
  return (
    <div className="border border-slate-300 rounded-lg p-4 text-center cursor-pointer hover:bg-slate-50" onClick={() => ref.current?.click()}>
      <p className="text-slate-600 text-sm">{label}</p>
      <input ref={ref} type="file" accept="image/*,video/*" className="hidden" onChange={e => {
        const f = e.target.files?.[0]; if (f) onFile(f)
      }} />
    </div>
  )
}