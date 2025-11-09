import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'
import Navbar from './components/Layout/Navbar'
import Sidebar from './components/Layout/Sidebar'
import Dashboard from './pages/Dashboard'
import Models from './pages/Models'
import Inference from './pages/Inference'
import PredictionForm from './pages/PredictionForm'
import Monitoring from './pages/Monitoring'
import Settings from './pages/Settings'
import TestingLab from './pages/TestingLab'
import { useState } from 'react'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Navbar toggleSidebar={toggleSidebar} />
      <Sidebar open={sidebarOpen} />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          mt: 8,
          ml: sidebarOpen ? '240px' : '0px',
          transition: 'margin-left 0.3s',
        }}
      >
        <Routes>
          <Route path="/" element={<TestingLab />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/models" element={<Models />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/prediction" element={<PredictionForm />} />
          <Route path="/monitoring" element={<Monitoring />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Box>
    </Box>
  )
}

export default App
