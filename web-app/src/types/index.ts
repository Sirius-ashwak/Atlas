export interface Device {
  id: string
  name: string
  type: 'sensor' | 'fog' | 'cloud'
  cpu: number
  memory: number
  bandwidth: number
  latency: number
  cost: number
}

export interface NetworkState {
  devices: Device[]
  num_devices: number
  num_fog_nodes: number
  num_cloud_nodes: number
}

export interface Allocation {
  device_id: string
  allocated_node: string
  confidence: number
}

export interface PredictionResponse {
  allocation: Allocation[]
  total_latency: number
  total_cost: number
  total_bandwidth: number
  inference_time: number
  model_used: string
}

export interface ModelInfo {
  name: string
  type: string
  path: string
  size_mb: number
  parameters: number
  loaded: boolean
  performance?: {
    reward: number
    training_steps: number
  }
}

export interface MetricsData {
  reward: number
  latency: number
  cost: number
  bandwidth: number
  timestamp: string
}

export interface HealthStatus {
  status: string
  version: string
  models_loaded: string[]
  uptime: number
}

export interface LabScenario {
  id: string
  title: string
  description: string
  params: {
    num_nodes: number
    num_edges: number
  }
  highlights: string[]
  recommendedModels: string[]
}

export interface LabRunResult {
  id: string
  model: string
  scenarioId: string
  scenarioTitle: string
  confidence: number
  selectedNode: number
  processingTimeMs: number
  nodeScores: Array<{ node: number; score: number }>
  runAt: string
  qValues?: number[]
}
