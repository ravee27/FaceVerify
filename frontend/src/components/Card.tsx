import React, { ReactNode } from 'react'

export default function Card({ title, children, action }: { title: string, children: ReactNode, action?: ReactNode }) {
  return (
    <div className="bg-white shadow-sm rounded-2xl p-5 border border-slate-200">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-slate-800">{title}</h2>
        {action}
      </div>
      {children}
    </div>
  )
}