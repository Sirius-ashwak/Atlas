import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { PredictionResponse } from '../../types'

interface PerformanceChartProps {
  predictions: PredictionResponse
}

const PerformanceChart: React.FC<PerformanceChartProps> = ({ predictions }) => {
  const data = [
    {
      name: 'Metrics',
      Latency: predictions.total_latency,
      Cost: predictions.total_cost * 10, // Scale for visibility
      Bandwidth: predictions.total_bandwidth,
    },
  ]

  return (
    <ResponsiveContainer width="100%" height="90%">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="Latency" fill="#1976d2" />
        <Bar dataKey="Cost" fill="#2e7d32" />
        <Bar dataKey="Bandwidth" fill="#ed6c02" />
      </BarChart>
    </ResponsiveContainer>
  )
}

export default PerformanceChart
