import { Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material'
import { useNavigate, useLocation } from 'react-router-dom'
import DashboardIcon from '@mui/icons-material/Dashboard'
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'
import ModelTrainingIcon from '@mui/icons-material/ModelTraining'
import PsychologyIcon from '@mui/icons-material/Psychology'
import ScienceIcon from '@mui/icons-material/Science'
import MonitorIcon from '@mui/icons-material/Monitor'
import SettingsIcon from '@mui/icons-material/Settings'

interface SidebarProps {
  open: boolean
}

const Sidebar: React.FC<SidebarProps> = ({ open }) => {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    { text: 'Testing Lab', icon: <AutoAwesomeIcon />, path: '/' },
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Models', icon: <ModelTrainingIcon />, path: '/models' },
    { text: 'Inference Chat', icon: <PsychologyIcon />, path: '/inference' },
    { text: 'Prediction Form', icon: <ScienceIcon />, path: '/prediction' },
    { text: 'Monitoring', icon: <MonitorIcon />, path: '/monitoring' },
  ]

  const settingsItems = [
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
  ]

  return (
    <Drawer
      variant="persistent"
      anchor="left"
      open={open}
      sx={{
        width: 240,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 240,
          boxSizing: 'border-box',
          mt: 8,
        },
      }}
    >
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <List>
        {settingsItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
    </Drawer>
  )
}

export default Sidebar
