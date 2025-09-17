export default function ScoreDial({ score, decision }: { score: number, decision: string }) {
  const color = decision === 'match' ? 'text-emerald-600' : 'text-rose-600'
  return (
    <div className="text-center">
      <div className={`text-6xl font-bold ${color}`}>{score}</div>
      <div className="uppercase tracking-wide text-slate-500">score (0–9999)</div>
      <div className={`mt-2 font-semibold ${color}`}>{decision}</div>
      <div className="mt-3 text-xs text-slate-500 max-w-sm mx-auto">
      </div>
    </div>
  )
}