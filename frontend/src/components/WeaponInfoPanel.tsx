import React from 'react'
import Card from './Card'
import Badge from './Badge'

export default function WeaponInfoPanel() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card title="Overview">
        <div className="space-y-2 text-sm text-slate-700">
          <div>Weapon detection identifies firearms/weapons in images and videos using an Inhouse Weapon Detection model.</div>
          <div className="flex flex-wrap gap-2 pt-1">
            <Badge variant="green">Real-time capable</Badge>
            <Badge variant="blue">GPU-accelerated</Badge>
            <Badge variant="amber">Confidence threshold configurable</Badge>
          </div>
        </div>
      </Card>
      <Card title="Model Info">
        <div className="space-y-2 text-sm text-slate-700">
          <div><span className="text-slate-500">Architecture:</span> YOLOv11</div>
          <div><span className="text-slate-500">Task:</span> Weapon detection</div>
          <div><span className="text-slate-500">Input:</span> Images, Videos, RTSP</div>
          <div><span className="text-slate-500">Output:</span> Bounding boxes + confidence</div>
          <div><span className="text-slate-500">Model Accuracy:</span> 91.5%</div>
        </div>
      </Card>
      <Card title="Accuracy Profile">
        <div className="space-y-2 text-sm text-slate-700">
          <div>Precision and recall depend on the chosen confidence threshold. Lower conf increases recall but may add false positives.</div>
          <div className="text-xs text-slate-500">Tip: Start at 0.40–0.50 and tune for your environment.</div>
        </div>
      </Card>
    </div>
  )
}


