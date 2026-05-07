import createPlotlyComponent from 'react-plotly.js/factory'
// 仅引入 basic 包，体积更小
// @ts-ignore  basic-dist-min 没有官方类型
import Plotly from 'plotly.js-basic-dist-min'

// 包装层：用更宽松的 props 类型，避免 plotly.js 严格类型与简写 title 字段的冲突
const RawPlot = createPlotlyComponent(Plotly)

interface PlotProps {
  data: any[]
  layout?: any
  config?: any
  style?: any
  useResizeHandler?: boolean
  className?: string
  onInitialized?: (...args: any[]) => void
  onUpdate?: (...args: any[]) => void
}

const Plot: (props: PlotProps) => any = RawPlot as any
export default Plot
