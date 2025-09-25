import React from 'react'
import Card from './Card'
import Badge from './Badge'

export default function ModelInfoPanel({ info }: { info: any }) {
  const device = info?.model?.device || '-'
  const arch = info?.model?.arch || '-'
  const torch = info?.model?.torch || '-'
  const source = info?.model?.state_dict ? 'state_dict' : (info?.model?.torchscript ? 'torchscript' : '-')

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card title="Overview">
        <div className="space-y-2 text-sm text-slate-700">
          <div>Face embeddings are compact numeric vectors that represent facial features.</div>
          <div>Similarity between two faces is measured via cosine similarity (scaled 0-9999).</div>
          <div className="flex flex-wrap gap-2 pt-1">
            <Badge variant="green">Real-time capable</Badge>
            <Badge variant="blue">GPU-accelerated</Badge>
            <Badge variant="amber">Privacy: no images stored by default</Badge>
          </div>
        </div>
      </Card>
      <Card title="Model Info">
        <div className="space-y-2 text-sm text-slate-700">
          <div><span className="text-slate-500">Device:</span> {device}</div>
          <div><span className="text-slate-500">Framework:</span> PyTorch {torch}</div>
          <div><span className="text-slate-500">Model source:</span> {source}</div>
          <div><span className="text-slate-500">Backbone:</span> {arch}</div>
          <div><span className="text-slate-500">Images Trained on:</span> 17M </div>
          <div><span className="text-slate-500">Model Accuracy:</span> 99.5% </div>
        </div>
      </Card>
      <Card title="Accuracy Profile">
        <div className="space-y-2 text-sm text-slate-700">
          <div>Thresholds balance false accepts vs false rejects. Profile <b>1in10k</b> targets a low False Accept Rate.</div>
          <div className="text-xs text-slate-500">Tip: For friendlier UX, you can relax thresholds when user supervision exists.</div>
        </div>
      </Card>
    </div>
  )
}


