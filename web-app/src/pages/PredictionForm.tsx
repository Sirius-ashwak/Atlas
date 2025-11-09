import { useState, useEffect } from 'react'
import {
  Typography,
  Box,
  Paper,
  Grid,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Slider,
  Card,
  CardContent,
  Alert,
  Chip,
  CircularProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material'
import {
  PlayArrow as RunIcon,
  Refresh as RefreshIcon,
  Science as TestIcon,
} from '@mui/icons-material'
import { toast } from 'react-toastify'
import api from '../services/api'
import { useAppStore } from '../store/useAppStore'

interface PredictionResult {
  allocated_node: string
  confidence: number
  metrics: {
    latency: number
    energy: number
    qos_score: number
  }
  model_used: string
  inference_time: number
}

const PredictionForm = () => {
  const { selectedModel, setSelectedModel } = useAppStore()
  
  // Form State
  const [numDevices, setNumDevices] = useState(5)
  const [numFog, setNumFog] = useState(3)
  const [numCloud, setNumCloud] = useState(2)
  const [networkLoad, setNetworkLoad] = useState(0.5)
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [models, setModels] = useState<any[]>([])

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const data = await api.listModels()
      setModels(data.models || [])
      if (!selectedModel && data.models.length > 0) {
        setSelectedModel(data.models[0].name)
      }
    } catch (error) {
      console.error('Failed to load models:', error)
    }
  }

  const handlePredict = async () => {
    if (!selectedModel) {
      toast.error('Please select a model first')
      return
    }

    setLoading(true)
    setPrediction(null)

    try {
      // Step 1: Generate mock network
      const totalNodes = numDevices + numFog + numCloud
      const mockNetwork = await api.generateMockNetwork({
        num_nodes: totalNodes,
        num_edges: Math.floor(totalNodes * 1.5),
      })

      // Step 2: Run prediction
      const result = await api.predict({
        model_type: selectedModel,
        network_state: mockNetwork.network_state,
      })

      setPrediction({
        allocated_node: result.allocation?.allocated_node || result.allocated_node || 'Unknown',
        confidence: result.allocation?.confidence || result.confidence || 0,
        metrics: {
          latency: result.metrics?.latency || 0,
          energy: result.metrics?.energy || 0,
          qos_score: result.metrics?.qos_score || 0,
        },
        model_used: selectedModel,
        inference_time: result.inference_time || 0,
      })

      toast.success('Prediction completed successfully!')
    } catch (error: any) {
      toast.error(`Prediction failed: ${error.message}`)
      console.error('Prediction error:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setNumDevices(5)
    setNumFog(3)
    setNumCloud(2)
    setNetworkLoad(0.5)
    setPrediction(null)
  }

  const totalNodes = numDevices + numFog + numCloud

  return (
    <Box>
      {/* Header with Gradient */}
      <Box 
        sx={{ 
          background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
          borderRadius: 3,
          p: 4,
          mb: 4,
          boxShadow: '0 10px 40px rgba(102, 126, 234, 0.3)',
        }}
      >
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h3" fontWeight="bold" color="white" gutterBottom>
              🎯 Model Prediction
            </Typography>
            <Typography variant="h6" color="rgba(255,255,255,0.9)">
              Configure network parameters and get instant AI predictions
            </Typography>
          </Box>
          <Chip
            label={selectedModel ? `${selectedModel}` : 'No Model'}
            color={selectedModel ? 'success' : 'default'}
            size="medium"
            sx={{ 
              fontSize: '1.1rem', 
              py: 3, 
              px: 2,
              fontWeight: 'bold',
              boxShadow: '0 4px 12px rgba(0,0,0,0.2)'
            }}
          />
        </Box>
      </Box>

      <Grid container spacing={3}>
        {/* Left Column: Input Form */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={8} 
            sx={{ 
              p: 4, 
              borderRadius: 3,
              background: 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.1)'
            }}
          >
            <Box display="flex" alignItems="center" gap={2} mb={3}>
              <Box 
                sx={{ 
                  background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                  borderRadius: 2,
                  p: 1.5,
                  display: 'flex'
                }}
              >
                <TestIcon sx={{ color: 'white', fontSize: 30 }} />
              </Box>
              <Typography variant="h5" fontWeight="bold" sx={{ 
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}>
                Network Configuration
              </Typography>
            </Box>
            <Divider sx={{ mb: 3 }} />

            {/* Model Selection */}
            <FormControl 
              fullWidth 
              sx={{ 
                mb: 3,
                '& .MuiOutlinedInput-root': {
                  borderRadius: 2,
                  backgroundColor: 'white',
                  boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }
              }}
            >
              <InputLabel>🤖 Select AI Model</InputLabel>
              <Select
                value={selectedModel || ''}
                onChange={(e) => setSelectedModel(e.target.value)}
                label="🤖 Select AI Model"
              >
                {models.map((model) => (
                  <MenuItem key={model.name} value={model.name}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Chip 
                        label={model.name} 
                        size="small" 
                        color="primary" 
                        variant="outlined" 
                      />
                      <Typography variant="body2" color="text.secondary">
                        {model.description?.split('-')[0] || model.type}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {/* Number of Devices */}
            <Box sx={{ mb: 4, p: 2, bgcolor: '#2d2d2d', borderRadius: 2 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body1" fontWeight="600" color="primary">
                  📱 IoT Devices
                </Typography>
                <Chip 
                  label={numDevices} 
                  color="primary" 
                  size="small"
                  sx={{ fontWeight: 'bold', fontSize: '1rem' }}
                />
              </Box>
              <Slider
                value={numDevices}
                onChange={(_, value) => setNumDevices(value as number)}
                min={1}
                max={20}
                marks
                valueLabelDisplay="auto"
                sx={{ 
                  mt: 1,
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 2px 8px rgba(102, 126, 234, 0.4)'
                  }
                }}
              />
            </Box>

            {/* Number of Fog Nodes */}
            <Box sx={{ mb: 4, p: 2, bgcolor: '#3d3d3d', borderRadius: 2 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body1" fontWeight="600" color="success.main">
                  🌫️ Fog Nodes
                </Typography>
                <Chip 
                  label={numFog} 
                  color="success" 
                  size="small"
                  sx={{ fontWeight: 'bold', fontSize: '1rem' }}
                />
              </Box>
              <Slider
                value={numFog}
                onChange={(_, value) => setNumFog(value as number)}
                min={1}
                max={10}
                marks
                valueLabelDisplay="auto"
                sx={{ 
                  mt: 1,
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 2px 8px rgba(46, 125, 50, 0.4)'
                  }
                }}
              />
            </Box>

            {/* Number of Cloud Nodes */}
            <Box sx={{ mb: 4, p: 2, bgcolor: '#4a4a4a', borderRadius: 2 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body1" fontWeight="600" color="warning.main">
                  ☁️ Cloud Nodes
                </Typography>
                <Chip 
                  label={numCloud} 
                  color="warning" 
                  size="small"
                  sx={{ fontWeight: 'bold', fontSize: '1rem' }}
                />
              </Box>
              <Slider
                value={numCloud}
                onChange={(_, value) => setNumCloud(value as number)}
                min={1}
                max={5}
                marks
                valueLabelDisplay="auto"
                sx={{ 
                  mt: 1,
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 2px 8px rgba(237, 108, 2, 0.4)'
                  }
                }}
              />
            </Box>

            {/* Network Load */}
            <Box sx={{ mb: 4, p: 2, bgcolor: '#5a5a5a', borderRadius: 2 }}>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="body1" fontWeight="600" color="error.main">
                  📊 Network Load
                </Typography>
                <Chip 
                  label={`${(networkLoad * 100).toFixed(0)}%`}
                  color="error" 
                  size="small"
                  sx={{ fontWeight: 'bold', fontSize: '1rem' }}
                />
              </Box>
              <Slider
                value={networkLoad}
                onChange={(_, value) => setNetworkLoad(value as number)}
                min={0}
                max={1}
                step={0.1}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${(value * 100).toFixed(0)}%`}
                sx={{ 
                  mt: 1,
                  '& .MuiSlider-thumb': {
                    boxShadow: '0 2px 8px rgba(211, 47, 47, 0.4)'
                  }
                }}
              />
            </Box>

            <Divider sx={{ my: 3 }} />

            {/* Summary */}
            <Paper 
              sx={{ 
                p: 3, 
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                color: 'white', 
                mb: 3,
                borderRadius: 2,
                boxShadow: '0 8px 24px rgba(102, 126, 234, 0.3)'
              }}
            >
              <Typography variant="h6" gutterBottom fontWeight="bold">
                📋 Summary
              </Typography>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body1">Total Nodes:</Typography>
                <Chip 
                  label={`${totalNodes} nodes`}
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 'bold' }}
                  size="small"
                />
              </Box>
              <Box display="flex" justifyContent="space-between" mb={1}>
                <Typography variant="body1">Configuration:</Typography>
                <Chip 
                  label={`${numDevices}D + ${numFog}F + ${numCloud}C`}
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 'bold' }}
                  size="small"
                />
              </Box>
              <Box display="flex" justifyContent="space-between">
                <Typography variant="body1">Estimated Edges:</Typography>
                <Chip 
                  label={Math.floor(totalNodes * 1.5)}
                  sx={{ bgcolor: 'rgba(255,255,255,0.2)', color: 'white', fontWeight: 'bold' }}
                  size="small"
                />
              </Box>
            </Paper>

            {/* Action Buttons */}
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="contained"
                fullWidth
                size="large"
                startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <RunIcon />}
                onClick={handlePredict}
                disabled={loading || !selectedModel}
                sx={{
                  py: 2,
                  fontSize: '1.1rem',
                  fontWeight: 'bold',
                  background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                  boxShadow: '0 6px 20px rgba(102, 126, 234, 0.4)',
                  '&:hover': {
                    background: 'linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%)',
                    boxShadow: '0 8px 30px rgba(102, 126, 234, 0.5)',
                    transform: 'translateY(-2px)',
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                {loading ? 'Predicting...' : '⚡ Run Prediction'}
              </Button>
              <Button
                variant="outlined"
                size="large"
                startIcon={<RefreshIcon />}
                onClick={handleReset}
                sx={{
                  py: 2,
                  borderWidth: 2,
                  '&:hover': {
                    borderWidth: 2,
                    transform: 'translateY(-2px)',
                  },
                  transition: 'all 0.3s ease'
                }}
              >
                Reset
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Right Column: Results */}
        <Grid item xs={12} md={6}>
          <Paper 
            elevation={8} 
            sx={{ 
              p: 4, 
              borderRadius: 3,
              background: 'linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%)',
              boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
              minHeight: '600px'
            }}
          >
            <Box display="flex" alignItems="center" gap={2} mb={3}>
              <Box 
                sx={{ 
                  background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                  borderRadius: 2,
                  p: 1.5,
                  display: 'flex'
                }}
              >
                <RunIcon sx={{ color: 'white', fontSize: 30 }} />
              </Box>
              <Typography variant="h5" fontWeight="bold" sx={{ 
                background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent'
              }}>
                Prediction Results
              </Typography>
            </Box>
            <Divider sx={{ mb: 3 }} />

            {!prediction && !loading && (
              <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                minHeight={400}
              >
                <TestIcon sx={{ fontSize: 80, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" color="text.secondary" textAlign="center">
                  Configure parameters and click "Run Prediction"
                </Typography>
                <Typography variant="body2" color="text.secondary" textAlign="center" mt={1}>
                  Results will appear here
                </Typography>
              </Box>
            )}

            {loading && (
              <Box
                display="flex"
                flexDirection="column"
                alignItems="center"
                justifyContent="center"
                minHeight={400}
              >
                <CircularProgress size={60} />
                <Typography variant="h6" mt={2}>
                  Running inference...
                </Typography>
              </Box>
            )}

            {prediction && !loading && (
              <Box>
                {/* Success Alert */}
                <Alert 
                  severity="success" 
                  sx={{ 
                    mb: 3,
                    borderRadius: 2,
                    boxShadow: '0 4px 12px rgba(46, 125, 50, 0.2)',
                    '& .MuiAlert-icon': {
                      fontSize: 28
                    }
                  }}
                >
                  <Typography variant="body1" fontWeight="600">
                    ✅ Prediction completed successfully using <strong>{prediction.model_used}</strong> model
                  </Typography>
                </Alert>

                {/* Main Result Card */}
                <Card 
                  sx={{ 
                    mb: 3, 
                    background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                    color: 'white',
                    borderRadius: 3,
                    boxShadow: '0 12px 40px rgba(17, 153, 142, 0.4)',
                    transform: 'scale(1.02)',
                    transition: 'transform 0.3s ease'
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Typography variant="h6" gutterBottom sx={{ opacity: 0.9 }}>
                      🎯 Recommended Node
                    </Typography>
                    <Typography 
                      variant="h2" 
                      fontWeight="bold" 
                      sx={{ 
                        my: 2,
                        textShadow: '0 2px 10px rgba(0,0,0,0.2)'
                      }}
                    >
                      {prediction.allocated_node}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="h5">
                        Confidence:
                      </Typography>
                      <Chip 
                        label={`${(prediction.confidence * 100).toFixed(1)}%`}
                        sx={{ 
                          bgcolor: 'rgba(255,255,255,0.3)', 
                          color: 'white',
                          fontSize: '1.2rem',
                          fontWeight: 'bold',
                          px: 2,
                          py: 3
                        }}
                      />
                    </Box>
                  </CardContent>
                </Card>

                {/* Metrics Grid */}
                <Grid container spacing={2} mb={3}>
                  <Grid item xs={4}>
                    <Card 
                      sx={{ 
                        background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                        color: 'white',
                        boxShadow: '0 4px 15px rgba(102, 126, 234, 0.3)',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: '0 8px 25px rgba(102, 126, 234, 0.4)',
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent>
                        <Typography sx={{ opacity: 0.9 }} variant="body2" gutterBottom>
                          ⚡ Latency
                        </Typography>
                        <Typography variant="h4" fontWeight="bold">
                          {prediction.metrics.latency.toFixed(2)}
                        </Typography>
                        <Typography variant="caption">milliseconds</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card 
                      sx={{ 
                        background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                        color: 'white',
                        boxShadow: '0 4px 15px rgba(240, 147, 251, 0.3)',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: '0 8px 25px rgba(240, 147, 251, 0.4)',
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent>
                        <Typography sx={{ opacity: 0.9 }} variant="body2" gutterBottom>
                          🔋 Energy
                        </Typography>
                        <Typography variant="h4" fontWeight="bold">
                          {prediction.metrics.energy.toFixed(2)}
                        </Typography>
                        <Typography variant="caption">units</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                  <Grid item xs={4}>
                    <Card 
                      sx={{ 
                        background: 'linear-gradient(135deg, #4a4a4a 0%, #5a5a5a 100%)',
                        color: 'white',
                        boxShadow: '0 4px 15px rgba(79, 172, 254, 0.3)',
                        '&:hover': {
                          transform: 'translateY(-4px)',
                          boxShadow: '0 8px 25px rgba(79, 172, 254, 0.4)',
                        },
                        transition: 'all 0.3s ease'
                      }}
                    >
                      <CardContent>
                        <Typography sx={{ opacity: 0.9 }} variant="body2" gutterBottom>
                          ⭐ QoS Score
                        </Typography>
                        <Typography variant="h4" fontWeight="bold">
                          {prediction.metrics.qos_score.toFixed(2)}
                        </Typography>
                        <Typography variant="caption">quality</Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>

                {/* Details Table */}
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell><strong>Metric</strong></TableCell>
                        <TableCell align="right"><strong>Value</strong></TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>Model Used</TableCell>
                        <TableCell align="right">{prediction.model_used}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Inference Time</TableCell>
                        <TableCell align="right">{(prediction.inference_time * 1000).toFixed(2)} ms</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Total Nodes</TableCell>
                        <TableCell align="right">{totalNodes}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>Network Configuration</TableCell>
                        <TableCell align="right">
                          {numDevices}D + {numFog}F + {numCloud}C
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>

                <Button
                  variant="contained"
                  fullWidth
                  size="large"
                  sx={{ 
                    mt: 3,
                    py: 2,
                    fontSize: '1.1rem',
                    fontWeight: 'bold',
                    background: 'linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)',
                    boxShadow: '0 6px 20px rgba(102, 126, 234, 0.4)',
                    '&:hover': {
                      background: 'linear-gradient(135deg, #2d2d2d 0%, #1a1a1a 100%)',
                      transform: 'translateY(-2px)',
                    },
                    transition: 'all 0.3s ease'
                  }}
                  onClick={handlePredict}
                  startIcon={<RunIcon />}
                >
                  🔄 Run Again
                </Button>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  )
}

export default PredictionForm
