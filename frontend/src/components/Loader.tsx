import React, { ReactNode } from 'react'

export default function Loader({ text = 'Loading…', dim = true }: { text?: string, dim?: boolean }) {
  return (
    <div className={`absolute inset-0 ${dim ? 'bg-white/60' : ''} flex items-center justify-center z-10`}>
      <div className="flex items-center gap-3 text-slate-700">
        <div className="h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm">{text}</span>
      </div>
    </div>
  )
}


