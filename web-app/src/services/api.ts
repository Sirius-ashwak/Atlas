import axios, { AxiosInstance, AxiosResponse } from 'axios'

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || '/api'

class ApiService {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error.response?.data || error.message)
        return Promise.reject(error)
      }
    )
  }

  // Health check
  async getHealth(): Promise<any> {
    const response: AxiosResponse = await this.client.get('/health')
    return response.data
  }

  // List available models
  async listModels(): Promise<any> {
    const response: AxiosResponse = await this.client.get('/models')
    return response.data
  }

  // Get model info
  async getModelInfo(modelName: string): Promise<any> {
    const response: AxiosResponse = await this.client.get(`/models/${modelName}`)
    return response.data
  }

  // Predict allocation
  async predict(data: any): Promise<any> {
    const response: AxiosResponse = await this.client.post('/predict', data)
    return response.data
  }

  // Batch predict
  async batchPredict(data: any): Promise<any> {
    const response: AxiosResponse = await this.client.post('/predict/batch', data)
    return response.data
  }

  // Get metrics
  async getMetrics(): Promise<any> {
    const response: AxiosResponse = await this.client.get('/metrics')
    return response.data
  }

  // Generate mock network
  async generateMockNetwork(params: any): Promise<any> {
    const response: AxiosResponse = await this.client.post('/generate-mock-network', params)
    return response.data
  }

  // Get training history
  async getTrainingHistory(modelName: string): Promise<any> {
    const response: AxiosResponse = await this.client.get(`/training-history/${modelName}`)
    return response.data
  }
}

export default new ApiService()
