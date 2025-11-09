import { useState, useRef, useEffect } from 'react'
import {
  Typography,
  Box,
  Paper,
  TextField,
  Button,
  Alert,
  CircularProgress,
  IconButton,
  Tooltip,
  Link as MuiLink,
} from '@mui/material'
import {
  Send as SendIcon,
  Psychology as AIIcon,
  Person as UserIcon,
  Delete as DeleteIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import api from '../services/api'
import { useAppStore } from '../store/useAppStore'
import { Link as RouterLink } from 'react-router-dom'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  data?: any
}

const Inference = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: '👋 Hello! I\'m your AI Edge Allocator assistant. I can help you predict optimal resource allocations for IoT networks.\n\nYou can:\n• Generate a mock network and get predictions\n• Ask me to analyze network states\n• Get allocation recommendations\n\nWhat would you like to do?',
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const { selectedModel } = useAppStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      // Analyze user input and decide action
      const lowerInput = input.toLowerCase()

      if (lowerInput.includes('generate') || lowerInput.includes('mock') || lowerInput.includes('network')) {
        // Generate mock network and predict
        const mockData = await api.generateMockNetwork({
          num_nodes: 10,
          num_edges: 15,
        })

        const prediction = await api.predict({
          model_type: selectedModel || 'hybrid',
          network_state: mockData.network_state,
        })

        const assistantMessage: Message = {
          role: 'assistant',
          content: `✅ Generated a mock IoT network and ran prediction!\n\n📊 **Results:**\n• Allocated Node: ${prediction.allocation.allocated_node}\n• Confidence: ${(prediction.allocation.confidence * 100).toFixed(1)}%\n• Latency: ${prediction.metrics.latency.toFixed(2)}ms\n• Energy: ${prediction.metrics.energy.toFixed(2)} units\n• QoS Score: ${prediction.metrics.qos_score.toFixed(2)}\n\n🔍 **Network Details:**\n• Total Nodes: ${mockData.network_state.nodes.length}\n• Total Edges: ${mockData.network_state.edges.length}\n• Model Used: ${selectedModel || 'hybrid'}`,
          timestamp: new Date(),
          data: prediction,
        }

        setMessages((prev) => [...prev, assistantMessage])
      } else if (lowerInput.includes('help') || lowerInput.includes('what can')) {
        // Help message
        const assistantMessage: Message = {
          role: 'assistant',
          content: `🤖 **I can help you with:**\n\n1. **Generate Mock Networks** - Say "generate a network" or "create mock data"\n2. **Run Predictions** - I'll analyze the network and suggest optimal allocations\n3. **View Metrics** - Check latency, energy, and QoS scores\n4. **Model Selection** - Use different AI models (DQN, PPO, Hybrid)\n\n💡 **Try asking:**\n• "Generate a network and predict"\n• "Create a mock IoT network"\n• "What's the best allocation?"\n• "Analyze this network"`,
          timestamp: new Date(),
        }

        setMessages((prev) => [...prev, assistantMessage])
      } else if (lowerInput.includes('model') || lowerInput.includes('change')) {
        const assistantMessage: Message = {
          role: 'assistant',
          content: `🤖 **Current Model:** ${selectedModel || 'None selected'}\n\nTo change models, go to the **Models** page (sidebar) and select a different model.\n\nAvailable models:\n• **Hybrid** - Best performance (DQN + PPO + GNN)\n• **DQN** - Value-based deep Q-learning\n• **PPO** - Policy-based proximal policy optimization`,
          timestamp: new Date(),
        }

        setMessages((prev) => [...prev, assistantMessage])
      } else {
        // Default: try to generate and predict
        const mockData = await api.generateMockNetwork({
          num_nodes: 10,
          num_edges: 15,
        })

        const prediction = await api.predict({
          model_type: selectedModel || 'hybrid',
          network_state: mockData.network_state,
        })

        const assistantMessage: Message = {
          role: 'assistant',
          content: `I understood you want predictions! Here's what I found:\n\n📊 **Allocation Result:**\n• Best Node: ${prediction.allocation.allocated_node}\n• Confidence: ${(prediction.allocation.confidence * 100).toFixed(1)}%\n\n📈 **Performance Metrics:**\n• Latency: ${prediction.metrics.latency.toFixed(2)}ms\n• Energy Cost: ${prediction.metrics.energy.toFixed(2)} units\n• QoS Score: ${prediction.metrics.qos_score.toFixed(2)}\n\nModel: ${selectedModel || 'hybrid'}`,
          timestamp: new Date(),
          data: prediction,
        }

        setMessages((prev) => [...prev, assistantMessage])
      }
    } catch (error: any) {
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Oops! Something went wrong:\n\n${error.message}\n\n💡 **Suggestions:**\n• Make sure the API is running (http://localhost:8000)\n• Check if a model is selected (go to Models page)\n• Verify the backend has trained models available`,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setMessages([
      {
        role: 'assistant',
        content: '👋 Chat cleared! How can I help you today?',
        timestamp: new Date(),
      },
    ])
  }

  return (
    <Box sx={{ height: 'calc(100vh - 200px)', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Box display="flex" alignItems="center">
          <AIIcon sx={{ fontSize: 40, mr: 2, color: 'primary.main' }} />
          <Box>
            <Typography variant="h4" fontWeight="bold">
              AI Inference Chat
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Interactive predictions powered by {selectedModel || 'Hybrid model'}
            </Typography>
          </Box>
        </Box>
        <Box>
          <Tooltip title="Clear chat">
            <IconButton onClick={handleClear} color="error">
              <DeleteIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Refresh">
            <IconButton onClick={() => window.location.reload()}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      <Alert severity="info" sx={{ mb: 2 }}>
        Looking for multi-model experimentation? Visit the{' '}
        <MuiLink component={RouterLink} to="/" underline="hover">
          Testing Lab
        </MuiLink>{' '}
        to compare allocation strategies side by side.
      </Alert>

      {!selectedModel && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          No model selected! Go to the <strong>Models</strong> page to select one, or I'll use the default Hybrid model.
        </Alert>
      )}

      {/* Chat Messages */}
      <Paper
        elevation={2}
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          p: 3,
          mb: 2,
          bgcolor: 'background.default',
        }}
      >
        {messages.map((message, index) => (
          <Box key={index} mb={3}>
            <Box display="flex" alignItems="center" mb={1}>
              {message.role === 'user' ? (
                <UserIcon sx={{ mr: 1, color: 'primary.main' }} />
              ) : (
                <AIIcon sx={{ mr: 1, color: 'secondary.main' }} />
              )}
              <Typography variant="subtitle2" fontWeight="bold">
                {message.role === 'user' ? 'You' : 'AI Assistant'}
              </Typography>
              <Typography variant="caption" color="text.secondary" ml={1}>
                {message.timestamp.toLocaleTimeString()}
              </Typography>
            </Box>

            <Paper
              elevation={1}
              sx={{
                p: 2,
                ml: 4,
                bgcolor: message.role === 'user' ? 'primary.light' : 'background.paper',
                color: message.role === 'user' ? 'primary.contrastText' : 'text.primary',
              }}
            >
              <Typography
                variant="body1"
                sx={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
              >
                {message.content}
              </Typography>
            </Paper>
          </Box>
        ))}

        {loading && (
          <Box display="flex" alignItems="center" ml={4}>
            <CircularProgress size={20} sx={{ mr: 2 }} />
            <Typography variant="body2" color="text.secondary">
              AI is thinking...
            </Typography>
          </Box>
        )}

        <div ref={messagesEndRef} />
      </Paper>

      {/* Input Area */}
      <Paper elevation={3} sx={{ p: 2 }}>
        <Box display="flex" gap={1}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder="Ask me to generate a network, predict allocations, or anything else..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
            disabled={loading}
          />
          <Button
            variant="contained"
            endIcon={<SendIcon />}
            onClick={handleSend}
            disabled={loading || !input.trim()}
            sx={{ minWidth: '120px' }}
          >
            Send
          </Button>
        </Box>
        <Typography variant="caption" color="text.secondary" mt={1} display="block">
          💡 Try: "Generate a network and predict" or "What's the best allocation?"
        </Typography>
      </Paper>
    </Box>
  )
}

export default Inference
