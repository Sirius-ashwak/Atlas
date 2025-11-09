import { Grid, Card, CardContent, Typography, Box } from '@mui/material'
import SpeedIcon from '@mui/icons-material/Speed'
import AccessTimeIcon from '@mui/icons-material/AccessTime'
import AttachMoneyIcon from '@mui/icons-material/AttachMoney'
import NetworkCheckIcon from '@mui/icons-material/NetworkCheck'
import { PredictionResponse } from '../../types'

interface MetricsCardsProps {
  predictions: PredictionResponse
}

const MetricsCards: React.FC<MetricsCardsProps> = ({ predictions }) => {
  const metrics = [
    {
      title: 'Total Latency',
      value: `${predictions.total_latency.toFixed(2)} ms`,
      icon: <AccessTimeIcon sx={{ fontSize: 40 }} />,
      color: '#1976d2',
    },
    {
      title: 'Total Cost',
      value: `$${predictions.total_cost.toFixed(2)}`,
      icon: <AttachMoneyIcon sx={{ fontSize: 40 }} />,
      color: '#2e7d32',
    },
    {
      title: 'Total Bandwidth',
      value: `${predictions.total_bandwidth.toFixed(2)} Mbps`,
      icon: <NetworkCheckIcon sx={{ fontSize: 40 }} />,
      color: '#ed6c02',
    },
    {
      title: 'Inference Time',
      value: `${predictions.inference_time.toFixed(3)} s`,
      icon: <SpeedIcon sx={{ fontSize: 40 }} />,
      color: '#9c27b0',
    },
  ]

  return (
    <Grid container spacing={3}>
      {metrics.map((metric, index) => (
        <Grid item xs={12} sm={6} md={3} key={index}>
          <Card sx={{ height: '100%', position: 'relative', overflow: 'visible' }}>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center">
                <Box>
                  <Typography color="textSecondary" gutterBottom variant="body2">
                    {metric.title}
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {metric.value}
                  </Typography>
                </Box>
                <Box
                  sx={{
                    bgcolor: metric.color,
                    color: 'white',
                    borderRadius: 2,
                    p: 1.5,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  {metric.icon}
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      ))}
    </Grid>
  )
}

export default MetricsCards
