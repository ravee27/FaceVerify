import React, { ReactNode } from 'react'

type Variant = 'blue' | 'green' | 'rose' | 'slate' | 'amber' | 'violet'

export default function Badge({ children, variant = 'slate' }: { children: ReactNode, variant?: Variant }) {
  const cls: Record<Variant, string> = {
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    rose: 'bg-rose-50 text-rose-700 border-rose-200',
    amber: 'bg-amber-50 text-amber-800 border-amber-200',
    violet: 'bg-violet-50 text-violet-700 border-violet-200',
  }
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full border ${cls[variant]}`}>
      {children}
    </span>
  )
}


