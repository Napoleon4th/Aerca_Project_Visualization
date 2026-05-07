import { useEffect, useState } from 'react'
import { useAppStore } from '../store'
import { createSession, getCausal, getSample, listDatasets } from '../api'
import type { DatasetInfo } from '../types'

export default function Sidebar() {
  const datasets = useAppStore((s) => s.datasets)
  const setDatasets = useAppStore((s) => s.setDatasets)
  const setSession = useAppStore((s) => s.setSession)
  const setSessionInfo = useAppStore((s) => s.setSessionInfo)
  const setCurrentSample = useAppStore((s) => s.setCurrentSample)
  const setCurrentSampleIdx = useAppStore((s) => s.setCurrentSampleIdx)
  const setTrueCausalMatrix = useAppStore((s) => s.setTrueCausalMatrix)
  const setRunStatus = useAppStore((s) => s.setRunStatus)
  const setResults = useAppStore((s) => s.setResults)
  const resetProgress = useAppStore((s) => s.resetProgress)
  const setToast = useAppStore((s) => s.setToast)

  const [datasetName, setDatasetName] = useState<string>('linear')
  const [adtype, setAdtype] = useState<string>('spike')
  const [preprocessing, setPreprocessing] = useState<number>(1)
  const [seed, setSeed] = useState<number>(42)
  const [trainingSize, setTrainingSize] = useState<number>(10)
  const [testingSize, setTestingSize] = useState<number>(10)
  const [T, setT] = useState<number>(200)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    listDatasets()
      .then((d: DatasetInfo[]) => setDatasets(d))
      .catch((e: any) => setToast({ kind: 'error', text: `加载数据集列表失败：${e.message}` }))
  }, [setDatasets, setToast])

  const datasetList = Array.isArray(datasets) ? datasets : []
  const currentDataset: DatasetInfo | undefined = datasetList.find((d) => d.name === datasetName)

  async function handleRun() {
    setBusy(true)
    try {
      const payload: any = {
        dataset_name: datasetName,
        preprocessing_data: preprocessing,
        seed,
        training_size: trainingSize,
        testing_size: testingSize,
        T,
      }
      if (currentDataset?.supports_adtype) payload.adtype = adtype

      const session = await createSession(payload)
      setSession(session)
      setSessionInfo({
        session_id: session.session_id,
        dataset_name: session.dataset_name,
        run_status: 'idle',
        n_total_samples: session.training_size + session.testing_size,
        training_size: session.training_size,
        testing_size: session.testing_size,
        use_slice: session.use_slice,
        supports_adtype: session.supports_adtype,
        num_vars: session.num_vars,
        T: session.T,
        has_causal_struct: session.has_causal_struct,
      })

      // 重置依赖于上一次会话的状态
      resetProgress()
      setRunStatus('idle')
      setResults(null)
      setCurrentSampleIdx(0)

      // 拉取首个样本和真实因果矩阵
      const sample = await getSample(session.session_id, 0, 'auto')
      setCurrentSample(sample)
      if (session.has_causal_struct) {
        const m = await getCausal(session.session_id)
        setTrueCausalMatrix(m)
      } else {
        setTrueCausalMatrix(null)
      }
      setToast({ kind: 'success', text: `数据准备完成 (session=${session.session_id.slice(0, 6)}...)` })
    } catch (e: any) {
      setToast({ kind: 'error', text: `数据准备失败：${e.message}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="w-64 shrink-0 glass-panel p-3 pb-4 overflow-y-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.14em] text-slate-500">Control</p>
          <h2 className="text-base font-semibold text-slate-800">🎛️ 数据配置</h2>
        </div>
        <span className="pill bg-slate-100 text-slate-700 border border-slate-200 text-[11px]">Session</span>
      </div>

      <div className="space-y-3">
        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">数据集</label>
          <select
            className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
          >
            {datasetList.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
              </option>
            ))}
          </select>
        </div>

        {currentDataset?.supports_adtype && (
          <div>
            <label className="block text-xs font-semibold text-slate-700 mb-1">异常类型</label>
            <select
              className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
              value={adtype}
              onChange={(e) => setAdtype(e.target.value)}
            >
              {currentDataset.adtypes.map((a) => (
                <option key={a} value={a}>
                  {a === 'spike' ? 'Spike（突然尖峰异常）' : a === 'step' ? 'Step（阶跃持续异常）' : a === 'causal' ? 'Causal Propagation（因果传播异常）' : a}
                </option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="block text-xs font-semibold text-slate-700 mb-1">数据处理方式</label>
          <div className="flex gap-2">
            <button
              type="button"
              className={`flex-1 text-xs py-1.5 rounded-lg border transition-all ${preprocessing === 1 ? 'bg-brand-500 text-white border-brand-500 shadow-sm' : 'bg-white text-slate-600 hover:border-brand-200'}`}
              onClick={() => setPreprocessing(1)}
            >
              生成新数据
            </button>
            <button
              type="button"
              className={`flex-1 text-xs py-1.5 rounded-lg border transition-all ${preprocessing === 0 ? 'bg-brand-500 text-white border-brand-500 shadow-sm' : 'bg-white text-slate-600 hover:border-brand-200'}`}
              onClick={() => setPreprocessing(0)}
            >
              加载已有数据
            </button>
          </div>
        </div>

        <div className="border border-slate-100 rounded-lg p-3 bg-white/70">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-slate-700">高级选项</span>
            <button
              type="button"
              className="text-xs text-brand-600 hover:underline"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? '收起' : '展开'}
            </button>
          </div>

          {showAdvanced && (
            <div className="mt-3 space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">训练样本</label>
                  <input
                    type="number"
                    className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    value={trainingSize}
                    min={1}
                    onChange={(e) => setTrainingSize(parseInt(e.target.value || '1', 10))}
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">测试样本</label>
                  <input
                    type="number"
                    className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    value={testingSize}
                    min={1}
                    onChange={(e) => setTestingSize(parseInt(e.target.value || '1', 10))}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">序列长度 T</label>
                  <input
                    type="number"
                    className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    value={T}
                    min={10}
                    onChange={(e) => setT(parseInt(e.target.value || '10', 10))}
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">随机种子</label>
                  <input
                    type="number"
                    className="w-full border rounded-lg px-3 py-2 text-sm bg-white/80 focus:border-brand-500 focus:ring-2 focus:ring-brand-100"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value || '0', 10))}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <button
          className="w-full mt-3 py-2.5 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white text-sm font-semibold shadow-lg shadow-emerald-200 hover:shadow-emerald-300 transition disabled:opacity-60"
          disabled={busy}
          onClick={handleRun}
        >
          {busy ? '⏳ 准备中…' : '🚀 准备数据集'}
        </button>
      </div>

      <p className="text-[10px] text-slate-400 mt-6">
        AERCA 多变量时间序列根因分析 · React + FastAPI
      </p>
    </aside>
  )
}
