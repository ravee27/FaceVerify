import React from 'react'

type Tab = { key: string; label: string }

export default function Tabs({ tabs, active, onChange }: { tabs: Tab[], active: string, onChange: (k: string) => void }) {
  return (
    <div className="flex gap-2 mb-6">
      {tabs.map(t => (
        <button key={t.key}
          onClick={() => onChange(t.key)}
          className={`px-4 py-2 rounded-full border ${active === t.key ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'}`}>
          {t.label}
        </button>
      ))}
    </div>
  )
}