import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material'
import { PredictionResponse } from '../../types'

interface AllocationTableProps {
  predictions: PredictionResponse
}

const AllocationTable: React.FC<AllocationTableProps> = ({ predictions }) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success'
    if (confidence >= 0.6) return 'warning'
    return 'error'
  }

  return (
    <TableContainer sx={{ maxHeight: 400 }}>
      <Table stickyHeader>
        <TableHead>
          <TableRow>
            <TableCell><strong>Device ID</strong></TableCell>
            <TableCell><strong>Allocated Node</strong></TableCell>
            <TableCell><strong>Confidence</strong></TableCell>
            <TableCell><strong>Status</strong></TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {predictions.allocation.map((alloc, index) => (
            <TableRow key={index} hover>
              <TableCell>{alloc.device_id}</TableCell>
              <TableCell>{alloc.allocated_node}</TableCell>
              <TableCell>
                <Chip
                  label={`${(alloc.confidence * 100).toFixed(1)}%`}
                  color={getConfidenceColor(alloc.confidence)}
                  size="small"
                />
              </TableCell>
              <TableCell>
                <Chip
                  label="Allocated"
                  color="primary"
                  size="small"
                  variant="outlined"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  )
}

export default AllocationTable
