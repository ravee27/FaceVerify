import React from 'react'
import Badge from './Badge'
import Card from './Card'

const Section = ({ title, children }: { title: string, children: React.ReactNode }) => (
  <div>
    <div className="flex items-center gap-2 mb-1">
      <h3 className="text-sm font-semibold text-slate-800">{title}</h3>
    </div>
    <div className="text-xs text-slate-700">{children}</div>
  </div>
)

export default function ApiDocs({ displayBase, mode }: { displayBase: string, mode: 'face' | 'weapon' }) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <Card title="Quick Start">
          <div className="text-sm text-slate-700 space-y-2">
            <div>Use these REST endpoints from your apps, scripts, or Postman.</div>
            <div className="flex items-center gap-2">
              <Badge variant="blue">Content-Type: multipart/form-data</Badge>
              <Badge variant="green">Auth: none</Badge>
              <Badge variant="amber">Latency depends on model size</Badge>
            </div>
          </div>
        </Card>
        <Card title="Base URL">
          <div className="text-sm text-slate-700">
            <code className="px-2 py-1 rounded bg-slate-100 border border-slate-200">{displayBase}/api</code>
          </div>
        </Card>
        <Card title="Health">
          <div className="text-sm text-slate-700">
            <div className="text-slate-500">GET {displayBase}/api/healthz</div>
            <pre className="mt-1 bg-slate-50 p-2 rounded border overflow-auto"><code>{`curl -s ${displayBase}/api/healthz`}</code></pre>
          </div>
        </Card>
      </div>

      {mode === 'face' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card title="Image vs Image">
            <div className="space-y-2 text-sm text-slate-700">
              <div className="flex items-center gap-2">
                <Badge variant="violet">POST</Badge>
                <code>{displayBase}/api/verify-images</code>
              </div>
              <div className="text-slate-600">Uploads two face images, detects faces, computes embeddings, and returns a similarity score.</div>
              <Section title="Form fields">
                <ul className="list-disc ml-5 space-y-1">
                  <li><b>img1</b> file — JPEG/PNG image</li>
                  <li><b>img2</b> file — JPEG/PNG image</li>
                  <li><b>keep_for_audit</b> boolean (optional)</li>
                  <li><b>retention_days</b> 15/30/45/90 (required if keep=true)</li>
                </ul>
              </Section>
              <Section title="Example">
                <pre className="bg-slate-50 p-2 rounded border overflow-auto"><code>{`curl -X POST ${displayBase}/api/verify-images \
  -F img1=@/path/a.jpg \
  -F img2=@/path/b.jpg \
  -F keep_for_audit=false \
  -F retention_days=0`}</code></pre>
              </Section>
            </div>
          </Card>
          <Card title="Video vs Reference">
            <div className="space-y-2 text-sm text-slate-700">
              <div className="flex items-center gap-2">
                <Badge variant="violet">POST</Badge>
                <code>{displayBase}/api/verify-video</code>
              </div>
              <div className="text-slate-600">Uploads a video and a single reference face image, and returns top matches with timestamps.</div>
              <Section title="Form fields">
                <ul className="list-disc ml-5 space-y-1">
                  <li><b>video</b> file — any video/*</li>
                  <li><b>ref_image</b> file — JPEG/PNG face image</li>
                  <li><b>keep_for_audit</b>, <b>retention_days</b> (optional)</li>
                </ul>
              </Section>
              <Section title="Example">
                <pre className="bg-slate-50 p-2 rounded border overflow-auto"><code>{`curl -X POST ${displayBase}/api/verify-video \
  -F video=@/path/clip.mp4 \
  -F ref_image=@/path/ref.jpg`}</code></pre>
              </Section>
            </div>
          </Card>
        </div>
      )}

      {mode === 'weapon' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card title="Weapon Detection (Image)">
            <div className="space-y-2 text-sm text-slate-700">
              <div className="flex items-center gap-2">
                <Badge variant="violet">POST</Badge>
                <code>{displayBase}/api/weapon/detect-image</code>
              </div>
              <div className="text-slate-600">Detects weapons in an uploaded image. Returns bounding boxes and confidence.</div>
              <Section title="Form fields">
                <ul className="list-disc ml-5 space-y-1">
                  <li><b>img</b> file — JPEG/PNG image</li>
                  <li><b>conf</b> number — confidence threshold (default 0.45)</li>
                </ul>
              </Section>
              <Section title="Example">
                <pre className="bg-slate-50 p-2 rounded border overflow-auto"><code>{`curl -X POST ${displayBase}/api/weapon/detect-image \
  -F img=@/path/image.jpg \
  -F conf=0.35`}</code></pre>
              </Section>
            </div>
          </Card>
          <Card title="Weapon Detection (Video)">
            <div className="space-y-2 text-sm text-slate-700">
              <div className="flex items-center gap-2">
                <Badge variant="violet">POST</Badge>
                <code>{displayBase}/api/weapon/detect-video</code>
              </div>
              <div className="text-slate-600">Queues a video for detection and streams annotated frames for preview.</div>
              <Section title="Form fields">
                <ul className="list-disc ml-5 space-y-1">
                  <li><b>video</b> file — any video/*</li>
                  <li><b>conf</b> number — confidence threshold</li>
                </ul>
              </Section>
              <Section title="Example">
                <pre className="bg-slate-50 p-2 rounded border overflow-auto"><code>{`curl -X POST ${displayBase}/api/weapon/detect-video \
  -F video=@/path/clip.mp4 \
  -F conf=0.35`}</code></pre>
              </Section>
            </div>
          </Card>
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card title="Errors & Tips">
          <ul className="list-disc ml-5 space-y-1 text-sm text-slate-700">
            <li>415 Unsupported Media Type — ensure you send multipart/form-data with the right fields.</li>
            <li>422 Unprocessable Entity — face not detected in image/video; try a clearer, frontal face.</li>
            <li>Latency depends on GPU availability; detector dominates processing time.</li>
          </ul>
        </Card>
        <Card title="SDK Snippet (JS)">
          <pre className="bg-slate-50 p-2 rounded border overflow-auto text-xs"><code>{`async function verify(a, b) {
  const fd = new FormData();
  fd.append('img1', a); fd.append('img2', b);
  const r = await fetch('${displayBase}/api/verify-images', { method: 'POST', body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}`}</code></pre>
        </Card>
      </div>
    </div>
  )
}


