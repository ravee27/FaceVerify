import React, { useEffect, useState } from 'react'
import Card from './components/Card'
import Tabs from './components/Tabs'
import UploadBox from './components/UploadBox'
import ScoreDial from './components/ScoreDial'
import StreamViewer from './components/StreamViewer'
import Toast from './components/Toast'
import Loader from './components/Loader'
import ApiDocs from './components/ApiDocs'
import ModelInfoPanel from './components/ModelInfoPanel'
import WeaponInfoPanel from './components/WeaponInfoPanel'

const API = (import.meta as any).env?.VITE_API_BASE || ''

type Mode = 'face' | 'weapon'

export default function App() {
  const [mode, setMode] = useState<Mode>('face')
  const [tab, setTab] = useState<'img'|'video'|'live'>('img')
  const [weaponTab, setWeaponTab] = useState<'img'|'video'|'live'>('img')
  const [toast, setToast] = useState<string>('')
  const [info, setInfo] = useState<any>(null)
  const [showApi, setShowApi] = useState(true)

  useEffect(() => {
    const load = async () => {
      try {
        const r = await fetch(`${API}/api/info`, { method: 'GET' })
        if (!r.ok) throw new Error('Failed to load service info')
        setInfo(await r.json())
      } catch (e:any) {
        setToast(e?.message || 'Failed to load service info')
      }
    }
    load()
  }, [])

  const displayBase = API || (typeof window !== 'undefined' ? window.location.origin : '')

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="mb-4 p-4 bg-white border border-slate-200 rounded-xl flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-slate-800">Computer Vision Toolkit</h1>
          <p className="text-slate-600 mt-1">Select a mode to get started.</p>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={()=>setMode('face')} className={`px-4 py-2 rounded-full border ${mode==='face'?'bg-blue-600 text-white border-blue-600':'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'}`}>Face Verify</button>
          <button onClick={()=>setMode('weapon')} className={`px-4 py-2 rounded-full border ${mode==='weapon'?'bg-blue-600 text-white border-blue-600':'bg-white text-slate-700 border-slate-300 hover:bg-slate-50'}`}>Weapon Detection</button>
        </div>
      </div>

      {mode==='face' && (
        <>
          <div className="mt-2">
            <Tabs tabs={[{key:'img',label:'Images'},{key:'video',label:'Video vs Ref'},{key:'live',label:'Live RTSP'}]} active={tab} onChange={(k)=>setTab(k as any)} />
          </div>

          <div className="mt-4 relative p-4 rounded-xl border border-slate-200 bg-slate-50 mb-6">
            {!info && <Loader text="Loading model & service info…" />}
            {info ? (
              <ModelInfoPanel info={info} />
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card title="Service Info"><div className="text-slate-400">Loading…</div></Card>
                <Card title="Model Info"><div className="text-slate-400">Loading…</div></Card>
                <Card title="Status"><div className="text-slate-400">Loading…</div></Card>
              </div>
            )}
          </div>

          {tab==='img' && <ImagesPanel setToast={setToast} />}
          {tab==='video' && <VideoPanel setToast={setToast} />}
          {tab==='live' && <LivePanel setToast={setToast} />}
        </>
      )}

      {mode==='weapon' && (
        <>
          <div className="mt-2">
            <Tabs tabs={[{key:'img',label:'Images'},{key:'video',label:'Video'},{key:'live',label:'Live RTSP'}]} active={weaponTab} onChange={(k)=>setWeaponTab(k as any)} />
          </div>
          <div className="mt-4 relative p-4 rounded-xl border border-slate-200 bg-slate-50 mb-6">
            <WeaponInfoPanel />
          </div>
          {weaponTab==='img' && <WeaponImagePanel setToast={setToast} />}
          {weaponTab==='video' && <WeaponVideoPanel setToast={setToast} />}
          {weaponTab==='live' && <WeaponLivePanel setToast={setToast} />}
        </>
      )}

      <div className="mt-8">
        <Card title="API Endpoints" action={<button className="text-sm px-3 py-1 border rounded" onClick={()=>setShowApi(v=>!v)}>{showApi? 'Hide' : 'Show'} details</button>}>
          {showApi && <ApiDocs displayBase={displayBase} mode={mode} />}
          {!showApi && <div className="text-sm text-slate-600">Click “Show details” for beautifully formatted API docs.</div>}
        </Card>
      </div>

      {toast && <Toast text={toast} onClose={()=>setToast('')} />}
    </div>
  )
}

function ImagesPanel({ setToast }: { setToast: (s:string)=>void }) {
  const [img1, setImg1] = useState<File | null>(null)
  const [img2, setImg2] = useState<File | null>(null)
  const [keep, setKeep] = useState(false)
  const [retention, setRetention] = useState(15)
  const [res, setRes] = useState<any>(null)
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!img1 || !img2) return setToast('Please select both images')
    setBusy(true)
    const fd = new FormData()
    fd.append('img1', img1)
    fd.append('img2', img2)
    fd.append('keep_for_audit', String(keep))
    fd.append('retention_days', String(keep ? retention : 0))
    try {
      const r = await fetch(`${API}/api/verify-images`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setRes(j)
    } catch (e:any) {
      setToast(e.message || 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative p-4 rounded-xl border-2 border-blue-200 bg-blue-50">
      {busy && <Loader text="Comparing images…" />}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Image 1">
          <UploadBox label={img1? img1.name : 'Click to upload'} onFile={setImg1} />
        </Card>
        <Card title="Image 2">
          <UploadBox label={img2? img2.name : 'Click to upload'} onFile={setImg2} />
        </Card>
        <Card title="Options">
          <div className="flex items-center gap-3">
            <input id="keep" type="checkbox" checked={keep} onChange={e=>setKeep(e.target.checked)} />
            <label htmlFor="keep">Keep for Audit</label>
            {keep && (
              <select className="ml-4 border rounded px-2 py-1" value={retention} onChange={e=>setRetention(Number(e.target.value))}>
                {[15,30,45,90].map(d=> <option key={d} value={d}>{d} days</option>)}
              </select>
            )}
          </div>
          <button className={`mt-4 px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} onClick={submit} disabled={busy}>{busy ? 'Processing…' : 'Compare'}</button>
        </Card>
        {res && (
          <Card title="Result">
            <div className="space-y-2">
              <ScoreDial score={res.similarity_scaled} decision={res.decision} />
              <div className="text-xs text-slate-500">The score 0-9999 is a scaled cosine similarity, with raw similarity ranging from -1 to 1.</div>
            </div>
          </Card>
        )}
      </div>
    </div>
  )
}

function VideoPanel({ setToast }: { setToast: (s:string)=>void }) {
  const [video, setVideo] = useState<File | null>(null)
  const [ref, setRef] = useState<File | null>(null)
  const [keep, setKeep] = useState(false)
  const [retention, setRetention] = useState(15)
  const [job, setJob] = useState<any>(null)
  const [topk, setTopk] = useState<any[]>([])
  const [busy, setBusy] = useState(false)

  const submit = async () => {
    if (!video || !ref) return setToast('Please select a video and a reference image')
    setBusy(true)
    const fd = new FormData()
    fd.append('video', video)
    fd.append('ref_image', ref)
    fd.append('keep_for_audit', String(keep))
    fd.append('retention_days', String(keep ? retention : 0))
    try {
      const r = await fetch(`${API}/api/verify-video`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      const j = await r.json()
      setJob(j)
      poll(j.job_id)
    } catch (e:any) {
      setToast(e.message || 'Request failed')
    } finally {
      setBusy(false)
    }
  }

  const poll = async (id: string) => {
    const t = setInterval(async () => {
      const r = await fetch(`${API}/api/jobs/${id}`)
      const j = await r.json()
      if (j.status === 'done') {
        clearInterval(t)
        setTopk(j.topk || [])
      }
    }, 1500)
  }

  return (
    <div className="relative p-4 rounded-xl border-2 border-blue-200 bg-blue-50">
      {busy && <Loader text="Submitting video…" />}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Video">
          <UploadBox label={video? video.name : 'Click to upload a video'} onFile={setVideo} />
        </Card>
        <Card title="Reference Image">
          <UploadBox label={ref? ref.name : 'Click to upload a face image'} onFile={setRef} />
        </Card>
        <Card title="Options">
          <div className="flex items-center gap-3">
            <input id="keepv" type="checkbox" checked={keep} onChange={e=>setKeep(e.target.checked)} />
            <label htmlFor="keepv">Keep for Audit</label>
            {keep && (
              <select className="ml-4 border rounded px-2 py-1" value={retention} onChange={e=>setRetention(Number(e.target.value))}>
                {[15,30,45,90].map(d=> <option key={d} value={d}>{d} days</option>)}
              </select>
            )}
          </div>
          <button className={`mt-4 px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} onClick={submit} disabled={busy}>{busy ? 'Uploading…' : 'Submit'}</button>
        </Card>
        {topk.length > 0 && (
          <Card title="Top Matches">
            <ul className="list-disc ml-5 text-slate-700">
              {topk.map((x,i)=> <li key={i}>t={x.t_sec.toFixed(2)}s — score {x.similarity_scaled}</li>)}
            </ul>
          </Card>
        )}
        {job?.job_id && (
          <div className="md:col-span-2">
            <Card title="Live Preview (annotated)">
              <div className="w-full flex justify-center">
                <img className="max-w-3xl w-full rounded border" src={`${API}/api/preview-video/${job.job_id}`} alt="preview" />
              </div>
              <div className="text-xs text-slate-500 mt-2">Green box indicates a match against the provided reference (threshold profile 1in10k).</div>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

function LivePanel({ setToast }: { setToast: (s:string)=>void }) {
  const [items, setItems] = useState<any[]>([])
  const [label, setLabel] = useState('')
  const [url, setUrl] = useState('')
  const [fps, setFps] = useState(3)
  const [selected, setSelected] = useState<string>('')

  const refresh = async () => {
    const r = await fetch(`${API}/api/streams`)
    const j = await r.json()
    setItems(j.items||[])
  }
  useEffect(()=>{ refresh() }, [])

  const add = async () => {
    if (!url) return setToast('Enter RTSP URL')
    const fd = new FormData()
    fd.append('rtsp_url', url)
    fd.append('label', label)
    fd.append('sampling_fps', String(fps))
    const r = await fetch(`${API}/api/streams`, { method: 'POST', body: fd })
    if (r.status === 409) {
      setToast('Max streams (5) reached. Delete one before adding.')
      return
    }
    if (!r.ok) { setToast('Failed to add stream'); return }
    setUrl(''); setLabel('');
    await refresh()
  }

  const delStream = async (id: string) => {
    await fetch(`${API}/api/streams/${id}`, { method: 'DELETE' })
    await refresh()
    if (selected === id) setSelected('')
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Card title="Add Stream">
        <div className="space-y-3">
          <input className="w-full border rounded px-3 py-2" placeholder="Label (optional)" value={label} onChange={e=>setLabel(e.target.value)} />
          <input className="w-full border rounded px-3 py-2" placeholder="rtsp://user:pass@host:port/…" value={url} onChange={e=>setUrl(e.target.value)} />
          <div className="flex items-center gap-2">
            <label>Sampling FPS</label>
            <input className="w-24 border rounded px-2 py-1" type="number" min={1} max={10} value={fps} onChange={e=>setFps(Number(e.target.value))} />
          </div>
          <button className="px-4 py-2 rounded bg-blue-600 text-white" onClick={add}>Add</button>
        </div>
      </Card>
      <Card title="Streams">
        <ul className="divide-y">
          {items.map((it:any)=> (
            <li key={it.stream_id} className="py-2 flex items-center justify-between">
              <div>
                <div className="font-medium">{it.label || it.stream_id.slice(0,8)}</div>
                <div className="text-sm text-slate-500">last score: {it.last_score ?? '-'}</div>
              </div>
              <div className="flex gap-2">
                <button className="px-3 py-1 border rounded" onClick={()=>setSelected(it.stream_id)}>View</button>
                <button className="px-3 py-1 border rounded text-rose-600" onClick={()=>delStream(it.stream_id)}>Delete</button>
              </div>
            </li>
          ))}
        </ul>
      </Card>
      <Card title="Live Scores">
        {selected ? <StreamViewer streamId={selected} /> : <div className="text-slate-500">Select a stream</div>}
      </Card>
    </div>
  )
}

function WeaponImagePanel({ setToast }: { setToast: (s:string)=>void }) {
  const [img, setImg] = useState<File | null>(null)
  const [res, setRes] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!img) return setToast('Select an image')
    setBusy(true)
    const fd = new FormData()
    fd.append('img', img)
    fd.append('conf', String(0.35))
    try {
      const r = await fetch(`${API}/api/weapon/detect-image`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      setRes(await r.json())
    } catch (e:any) { setToast(e.message || 'Request failed') }
    finally { setBusy(false) }
  }
  return (
    <div className="relative p-4 rounded-xl border-2 border-emerald-200 bg-emerald-50">
      {busy && <Loader text="Detecting weapon…" />}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card title="Image">
          <div className="space-y-4">
            <UploadBox label={img? img.name : 'Click to upload'} onFile={setImg} />
            <button className={`px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} disabled={busy} onClick={submit}>{busy ? 'Processing…' : 'Detect Weapon'}</button>
            <div className="text-xs text-slate-500">Upload an image and click Detect to see bounding boxes and confidence.</div>
          </div>
        </Card>
        <div className="md:col-span-2">
          <Card title="Detection Result" action={res?.items?.length ? <span className="text-xs px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200">{res.items.length} detections</span> : undefined}>
            <div className="w-full flex justify-center">
              {res?.preview_jpeg_b64 ? (
                <img className="max-w-4xl w-full rounded border" src={`data:image/jpeg;base64,${res.preview_jpeg_b64}`} />
              ) : (
                <div className="text-slate-500 text-sm p-6 w-full text-center">No result yet</div>
              )}
            </div>
            <div className="text-sm text-slate-700 mt-4">
              <div className="font-semibold mb-2">Detections</div>
              {Array.isArray(res?.items) && res.items.length ? (
                <ul className="list-disc ml-5 space-y-1">
                  {res.items.map((x:any,i:number)=> <li key={i}>[{x.bbox.join(', ')}] — {(x.score*100).toFixed(0)}%</li>)}
                </ul>
              ) : <div className="text-slate-500">None</div>}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function WeaponVideoPanel({ setToast }: { setToast: (s:string)=>void }) {
  const [video, setVideo] = useState<File | null>(null)
  const [job, setJob] = useState<any>(null)
  const [busy, setBusy] = useState(false)
  const submit = async () => {
    if (!video) return setToast('Select a video')
    setBusy(true)
    const fd = new FormData()
    fd.append('video', video)
    fd.append('conf', String(0.35))
    try {
      const r = await fetch(`${API}/api/weapon/detect-video`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error(await r.text())
      setJob(await r.json())
    } catch (e:any) { setToast(e.message || 'Request failed') }
    finally { setBusy(false) }
  }
  return (
    <div className="relative p-4 rounded-xl border-2 border-emerald-200 bg-emerald-50">
      {busy && <Loader text="Uploading video…" />}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Card title="Video">
          <div className="space-y-4">
            <UploadBox label={video? video.name : 'Click to upload a video'} onFile={setVideo} />
            <button className={`px-4 py-2 rounded bg-blue-600 text-white ${busy?'opacity-50':''}`} disabled={busy} onClick={submit}>{busy ? 'Processing…' : 'Detect Weapon'}</button>
            <div className="text-xs text-slate-500">Upload a clip and view live annotated results on the right.</div>
          </div>
        </Card>
        <div className="md:col-span-2">
          <Card title="Detection Result / Live Preview">
            <div className="w-full flex justify-center min-h-48">
              {job?.job_id ? (
                <img className="max-w-4xl w-full rounded border" src={`${API}/api/weapon/preview-video/${job.job_id}`} />
              ) : (
                <div className="text-slate-500 text-sm p-6">Upload a video to start detection</div>
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function WeaponLivePanel({ setToast }: { setToast: (s:string)=>void }) {
  const [url, setUrl] = useState('')
  const [conf, setConf] = useState(0.35)
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card title="RTSP Source">
        <div className="space-y-3">
          <input className="w-full border rounded px-3 py-2" placeholder="rtsp://user:pass@host:port/…" value={url} onChange={e=>setUrl(e.target.value)} />
          <div className="flex items-center gap-2">
            <label>Conf</label>
            <input className="w-24 border rounded px-2 py-1" type="number" min={0.1} max={0.8} step={0.05} value={conf} onChange={e=>setConf(Number(e.target.value))} />
          </div>
        </div>
      </Card>
      <Card title="Live Preview (annotated)">
        {url ? (
          <div className="w-full flex justify-center">
            <img className="max-w-4xl w-full rounded border" src={`${API}/api/weapon/preview-rtsp?rtsp_url=${encodeURIComponent(url)}&conf=${conf}`} />
          </div>
        ) : <div className="text-slate-500">Enter RTSP URL to start</div>}
      </Card>
    </div>
  )
}