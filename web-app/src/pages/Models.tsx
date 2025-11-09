import { useState, useEffect } from 'react'
import {
  Typography,
  Box,
  Card,
  CardContent,
  CardActions,
  Button,
  Grid,
  Chip,
  CircularProgress,
  Alert,
  Paper,
  Divider,
} from '@mui/material'
import {
  SmartToy as ModelIcon,
  CheckCircle as CheckIcon,
  Info as InfoIcon,
  PlayArrow as PlayIcon,
} from '@mui/icons-material'
import api from '../services/api'
import { useAppStore } from '../store/useAppStore'

interface ModelInfo {
  name: string
  type: string
  description: string
  path: string
  status: string
  performance?: {
    mean_reward: number
    std_reward: number
  }
}

const Models = () => {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { selectedModel, setSelectedModel } = useAppStore()

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await api.listModels()
      setModels(response.models || [])
    } catch (err: any) {
      setError(err.message || 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }

  const handleSelectModel = (modelName: string) => {
    setSelectedModel(modelName)
  }

  const getModelColor = (type: string) => {
    switch (type.toLowerCase()) {
      case 'hybrid':
        return 'primary'
      case 'dqn':
        return 'secondary'
      case 'ppo':
        return 'success'
      default:
        return 'default'
    }
  }

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    )
  }

  if (error) {
    return (
      <Box>
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
        <Button variant="contained" onClick={loadModels}>
          Retry
        </Button>
      </Box>
    )
  }

  return (
    <Box>
      <Box display="flex" alignItems="center" mb={3}>
        <ModelIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
        <Box>
          <Typography variant="h4" fontWeight="bold">
            Model Selection
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Choose a trained model for inference and predictions
          </Typography>
        </Box>
      </Box>

      {selectedModel && (
        <Alert
          severity="success"
          icon={<CheckIcon />}
          sx={{ mb: 3 }}
          action={
            <Button size="small" onClick={() => setSelectedModel('')}>
              Clear
            </Button>
          }
        >
          Currently selected: <strong>{selectedModel}</strong>
        </Alert>
      )}

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} md={6} lg={4} key={model.name}>
            <Card
              elevation={selectedModel === model.name ? 8 : 2}
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                border: selectedModel === model.name ? 2 : 0,
                borderColor: 'primary.main',
                transition: 'all 0.3s',
                '&:hover': {
                  elevation: 6,
                  transform: 'translateY(-4px)',
                },
              }}
            >
              <CardContent sx={{ flexGrow: 1 }}>
                <Box display="flex" alignItems="center" mb={2}>
                  <ModelIcon sx={{ mr: 1, color: 'primary.main' }} />
                  <Typography variant="h6" fontWeight="bold">
                    {model.name}
                  </Typography>
                </Box>

                <Box mb={2}>
                  <Chip
                    label={model.type.toUpperCase()}
                    color={getModelColor(model.type)}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  <Chip
                    label={model.status}
                    color={model.status === 'available' ? 'success' : 'warning'}
                    size="small"
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" mb={2}>
                  {model.description}
                </Typography>

                {model.performance && (
                  <Paper variant="outlined" sx={{ p: 1.5, bgcolor: 'background.default' }}>
                    <Typography variant="caption" color="text.secondary" display="block">
                      Performance
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      Reward: {model.performance.mean_reward.toFixed(2)} ± {model.performance.std_reward.toFixed(2)}
                    </Typography>
                  </Paper>
                )}
              </CardContent>

              <Divider />

              <CardActions sx={{ justifyContent: 'space-between', p: 2 }}>
                <Button
                  size="small"
                  startIcon={<InfoIcon />}
                  onClick={() => alert(`Model: ${model.name}\nPath: ${model.path}`)}
                >
                  Details
                </Button>
                <Button
                  variant={selectedModel === model.name ? 'contained' : 'outlined'}
                  size="small"
                  startIcon={selectedModel === model.name ? <CheckIcon /> : <PlayIcon />}
                  onClick={() => handleSelectModel(model.name)}
                >
                  {selectedModel === model.name ? 'Selected' : 'Select'}
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      {models.length === 0 && (
        <Alert severity="info" icon={<InfoIcon />}>
          No models available. Please train a model first.
        </Alert>
      )}
    </Box>
  )
}

export default Models
