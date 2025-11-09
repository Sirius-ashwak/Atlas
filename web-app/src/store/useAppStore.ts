import { create } from 'zustand'
import { NetworkState, PredictionResponse, ModelInfo, MetricsData } from '../types'

interface AppState {
  // Network state
  networkState: NetworkState | null
  setNetworkState: (state: NetworkState) => void

  // Predictions
  predictions: PredictionResponse | null
  setPredictions: (predictions: PredictionResponse) => void

  // Models
  availableModels: ModelInfo[]
  setAvailableModels: (models: ModelInfo[]) => void
  selectedModel: string
  setSelectedModel: (model: string) => void

  // Metrics history
  metricsHistory: MetricsData[]
  addMetrics: (metrics: MetricsData) => void
  clearMetrics: () => void

  // Loading states
  loading: boolean
  setLoading: (loading: boolean) => void

  // Error handling
  error: string | null
  setError: (error: string | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  // Network state
  networkState: null,
  setNetworkState: (networkState) => set({ networkState }),

  // Predictions
  predictions: null,
  setPredictions: (predictions) => set({ predictions }),

  // Models
  availableModels: [],
  setAvailableModels: (availableModels) => set({ availableModels }),
  selectedModel: 'hybrid',
  setSelectedModel: (selectedModel) => set({ selectedModel }),

  // Metrics history
  metricsHistory: [],
  addMetrics: (metrics) =>
    set((state) => ({
      metricsHistory: [...state.metricsHistory, metrics].slice(-50), // Keep last 50
    })),
  clearMetrics: () => set({ metricsHistory: [] }),

  // Loading states
  loading: false,
  setLoading: (loading) => set({ loading }),

  // Error handling
  error: null,
  setError: (error) => set({ error }),
}))
