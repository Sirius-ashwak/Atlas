import { AppBar, Toolbar, Typography, IconButton, Box } from '@mui/material'
import MenuIcon from '@mui/icons-material/Menu'
import DashboardIcon from '@mui/icons-material/Dashboard'

interface NavbarProps {
  toggleSidebar: () => void
}

const Navbar: React.FC<NavbarProps> = ({ toggleSidebar }) => {
  return (
    <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
      <Toolbar>
        <IconButton
          color="inherit"
          aria-label="toggle sidebar"
          edge="start"
          onClick={toggleSidebar}
          sx={{ mr: 2 }}
        >
          <MenuIcon />
        </IconButton>
        <DashboardIcon sx={{ mr: 1 }} />
        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
          AI Edge Allocator
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="body2" sx={{ opacity: 0.8 }}>
            Hybrid DQN-PPO-GNN
          </Typography>
        </Box>
      </Toolbar>
    </AppBar>
  )
}

export default Navbar
