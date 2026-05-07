import { useMemo } from 'react'
import Plot from './Plot'
import { useAppStore } from '../store'

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

function getVarNames(dataset: string, n: number): string[] {
  if (dataset === 'lotka_volterra') {
    const p = Math.floor(n / 2)
    return [
      ...Array.from({ length: p }, (_, i) => `Prey_${i}`),
      ...Array.from({ length: n - p }, (_, i) => `Predator_${i}`),
    ]
  }
  if (dataset === 'lorenz96') return Array.from({ length: n }, (_, i) => `X_${i}`)
  return Array.from({ length: n }, (_, i) => `var_${i}`)
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

export default function ResultsView() {
  const session = useAppStore((s) => s.session)
  const sample = useAppStore((s) => s.currentSample)
  const idx = useAppStore((s) => s.currentSampleIdx)
  const results = useAppStore((s) => s.results)

  const varNames = useMemo(
    () => (session ? getVarNames(session.dataset_name, session.num_vars) : []),
    [session],
  )

  if (!results || !session) return null

  const rc = results.root_cause
  const cd = results.causal_discovery

  // 当前样本根因预测
  const predicted = rc.predicted_root_causes.find((p) => p.sample_idx === idx)

  // 异常区间（从 label 中提取）
  let anomalyShapes: any[] = []
  if (sample) {
    const anomalyMask = sample.label.map((row) => row.some((v) => v > 0))
    let inAnomaly = false
    let start = 0
    for (let t = 0; t < anomalyMask.length; t++) {
      if (anomalyMask[t] && !inAnomaly) {
        start = t
        inAnomaly = true
      } else if (!anomalyMask[t] && inAnomaly) {
        anomalyShapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: start,
          x1: t,
          y0: 0,
          y1: 1,
          fillcolor: 'orange',
          opacity: 0.2,
          line: { width: 0 },
        })
        inAnomaly = false
      }
    }
    if (inAnomaly) {
      anomalyShapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: start,
        x1: anomalyMask.length,
        y0: 0,
        y1: 1,
        fillcolor: 'orange',
        opacity: 0.2,
        line: { width: 0 },
      })
    }
  }

  // 根因预测高亮（红色虚线）
  if (predicted && sample) {
    anomalyShapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: predicted.root_cause_time,
      x1: predicted.root_cause_time,
      y0: 0,
      y1: 1,
      line: { color: 'red', width: 3, dash: 'dash' },
    })
  }

  const highlightTraces = sample
    ? sample.x_ab[0].map((_, v) => ({
        x: Array.from({ length: sample.T }, (_, i) => i),
        y: sample.x_ab.map((row) => row[v]),
        mode: 'lines',
        name: varNames[v],
        line: { color: COLORS[v % COLORS.length], width: 1.5 },
      }))
    : []

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4 space-y-4">
        <h3 className="section-title">🎯 根因分析效果一览</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="metric-card">
            <span className="metric-label">变量定位准确率 (Top-5)</span>
            <span className="metric-value">{pct(rc.ac_at[2] ?? 0)}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">联合准确率 (Top-10)</span>
            <span className="metric-value">{pct(rc.ac_star_at[1] ?? 0)}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Top-1 准确率</span>
            <span className="metric-value">{pct(rc.ac_at[0] ?? 0)}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">Top-10 平均</span>
            <span className="metric-value">{rc.avg_at_10.toFixed(3)}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">严格联合 Top-1</span>
            <span className="metric-value">{pct(rc.ac_star_at[0] ?? 0)}</span>
          </div>
          <div className="metric-card">
            <span className="metric-label">严格联合 Top-100</span>
            <span className="metric-value">{pct(rc.ac_star_at[2] ?? 0)}</span>
          </div>
        </div>
      </div>

      {sample && predicted && (
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="section-title">异常时序 + 根因高亮（样本 #{idx}）</h3>
          <p className="text-xs text-slate-500 mb-2">
            <span className="inline-block w-3 h-3 bg-orange-300 mr-1 align-middle" />
            真实异常区间 ·
            <span className="inline-block w-3 h-0.5 bg-red-500 mx-1 align-middle border-dashed" />
            模型预测根因：{varNames[predicted.root_cause_var_idx]} @ t={predicted.root_cause_time}
          </p>
          <Plot
            data={highlightTraces as any}
            layout={{
              height: 460,
              xaxis: { title: '时间步' },
              yaxis: { title: '变量值' },
              shapes: anomalyShapes,
              margin: { t: 30, b: 50, l: 60, r: 30 },
              legend: { title: { text: '变量' } },
            }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      )}

      {cd && (
        <div className="bg-white rounded-lg shadow p-4 space-y-4">
          <h3 className="section-title">🔗 因果发现结果</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="metric-card">
              <span className="metric-label">F1 分数</span>
              <span className="metric-value">{cd.f1_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">AUROC</span>
              <span className="metric-value">{cd.auroc_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">AUPRC</span>
              <span className="metric-value">{cd.auprc_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Hamming Distance</span>
              <span className="metric-value">{cd.hamming_mean.toFixed(4)}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div>
              <p className="text-sm font-medium text-slate-700 mb-1">真实因果矩阵</p>
              <Plot
                data={[
                  {
                    z: cd.true_causal_matrix,
                    x: varNames,
                    y: varNames,
                    type: 'heatmap',
                    colorscale: 'Blues',
                  } as any,
                ]}
                layout={{ height: 400, yaxis: { autorange: 'reversed' }, margin: { t: 30, b: 50, l: 80, r: 30 } }}
                style={{ width: '100%' }}
                useResizeHandler
                config={{ responsive: true, displayModeBar: false }}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 mb-1">模型预测因果矩阵</p>
              <Plot
                data={[
                  {
                    z: cd.predicted_causal_matrix,
                    x: varNames,
                    y: varNames,
                    type: 'heatmap',
                    colorscale: 'Blues',
                  } as any,
                ]}
                layout={{ height: 400, yaxis: { autorange: 'reversed' }, margin: { t: 30, b: 50, l: 80, r: 30 } }}
                style={{ width: '100%' }}
                useResizeHandler
                config={{ responsive: true, displayModeBar: false }}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
