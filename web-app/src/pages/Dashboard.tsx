import { useEffect, useState } from 'react'
import { Grid, Paper, Typography, Box, Button, CircularProgress } from '@mui/material'
import { toast } from 'react-toastify'
import RefreshIcon from '@mui/icons-material/Refresh'
import NetworkTopology from '../components/Dashboard/NetworkTopology'
import MetricsCards from '../components/Dashboard/MetricsCards'
import AllocationTable from '../components/Dashboard/AllocationTable'
import PerformanceChart from '../components/Dashboard/PerformanceChart'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'

const Dashboard = () => {
  const {
    networkState,
    setNetworkState,
    predictions,
    setPredictions,
    selectedModel,
    loading,
    setLoading,
  } = useAppStore()

  const [health, setHealth] = useState<any>(null)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // Load health status
      const healthData = await ApiService.getHealth()
      setHealth(healthData)

      // Generate mock network if none exists
      if (!networkState) {
        const mockNetwork = await ApiService.generateMockNetwork({
          num_devices: 15,
          num_fog: 3,
          num_cloud: 2,
        })
        setNetworkState(mockNetwork)

        // Get initial prediction
        const prediction = await ApiService.predict({
          network_state: mockNetwork,
          model_name: selectedModel,
        })
        setPredictions(prediction)
      }

      toast.success('Dashboard loaded successfully')
    } catch (error: any) {
      toast.error(`Failed to load dashboard: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleRefresh = async () => {
    await loadData()
  }

  const handleRunInference = async () => {
    if (!networkState) return
    
    setLoading(true)
    try {
      const prediction = await ApiService.predict({
        network_state: networkState,
        model_name: selectedModel,
      })
      setPredictions(prediction)
      toast.success('Inference completed successfully')
    } catch (error: any) {
      toast.error(`Inference failed: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  if (loading && !networkState) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress size={60} />
      </Box>
    )
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" fontWeight="bold">
          Dashboard
        </Typography>
        <Box display="flex" gap={2}>
          <Button
            variant="contained"
            startIcon={<RefreshIcon />}
            onClick={handleRefresh}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            color="secondary"
            onClick={handleRunInference}
            disabled={loading || !networkState}
          >
            Run Inference
          </Button>
        </Box>
      </Box>

      {/* System Status */}
      {health && (
        <Paper sx={{ p: 2, mb: 3, bgcolor: 'success.light', color: 'white' }}>
          <Typography variant="h6">
            System Status: {health.status} | Version: {health.version}
          </Typography>
          <Typography variant="body2">
            Models Loaded: {
              health.models_loaded 
                ? Object.entries(health.models_loaded)
                    .filter(([_, loaded]) => loaded)
                    .map(([name]) => name)
                    .join(', ')
                : 'None'
            }
          </Typography>
        </Paper>
      )}

      {/* Metrics Cards */}
      {predictions && (
        <Grid container spacing={3} mb={3}>
          <Grid item xs={12}>
            <MetricsCards predictions={predictions} />
          </Grid>
        </Grid>
      )}

      {/* Main Visualizations */}
      <Grid container spacing={3}>
        {/* Network Topology */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2, height: '500px' }}>
            <Typography variant="h6" mb={2}>
              Network Topology
            </Typography>
            {networkState && (
              <NetworkTopology 
                networkState={networkState} 
                predictions={predictions}
              />
            )}
          </Paper>
        </Grid>

        {/* Performance Chart */}
        <Grid item xs={12} lg={6}>
          <Paper sx={{ p: 2, height: '500px' }}>
            <Typography variant="h6" mb={2}>
              Performance Metrics
            </Typography>
            {predictions && <PerformanceChart predictions={predictions} />}
          </Paper>
        </Grid>

        {/* Allocation Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" mb={2}>
              Device Allocations
            </Typography>
            {predictions && <AllocationTable predictions={predictions} />}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default Dashboard
