import React, { useEffect, useRef, useState } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export default function StreamViewer({ streamId }: { streamId: string }) {
  const [data, setData] = useState<{ t: number; score: number }[]>([])
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    const ws = new WebSocket(`${location.origin.replace('http', 'ws')}/api/ws/streams/${streamId}`)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data)
      if (msg.event === 'score') {
        setData(prev => {
          const next = [...prev, { t: msg.t, score: msg.score }]
          return next.slice(-100)
        })
      }
    }
    ws.onclose = () => { /* noop */ }
    return () => ws.close()
  }, [streamId])

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <XAxis dataKey="t" tickFormatter={(v) => new Date(v * 1000).toLocaleTimeString()} hide={true} />
          <YAxis domain={[0, 9999]} />
          <Tooltip labelFormatter={(v) => new Date(Number(v) * 1000).toLocaleTimeString()} />
          <Line type="monotone" dataKey="score" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}