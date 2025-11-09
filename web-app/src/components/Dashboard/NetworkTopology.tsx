import { useEffect, useRef } from 'react'
import { Box } from '@mui/material'
import * as d3 from 'd3'
import { NetworkState, PredictionResponse } from '../../types'

interface NetworkTopologyProps {
  networkState: NetworkState
  predictions: PredictionResponse | null
}

const NetworkTopology: React.FC<NetworkTopologyProps> = ({ networkState, predictions }) => {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !networkState) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    const width = 600
    const height = 400
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Create nodes data
    const nodes = networkState.devices.map((device) => ({
      id: device.id,
      name: device.name,
      type: device.type,
      x: Math.random() * (width - 100) + 50,
      y: Math.random() * (height - 100) + 50,
    }))

    // Create links from predictions
    const links: any[] = []
    if (predictions) {
      predictions.allocation.forEach((alloc) => {
        const source = nodes.find((n) => n.id === alloc.device_id)
        const target = nodes.find((n) => n.id === alloc.allocated_node)
        if (source && target) {
          links.push({ source: source.id, target: target.id })
        }
      })
    }

    // Draw links
    g.selectAll('.link')
      .data(links)
      .enter()
      .append('line')
      .attr('class', 'link')
      .attr('x1', (d: any) => nodes.find((n) => n.id === d.source)?.x || 0)
      .attr('y1', (d: any) => nodes.find((n) => n.id === d.source)?.y || 0)
      .attr('x2', (d: any) => nodes.find((n) => n.id === d.target)?.x || 0)
      .attr('y2', (d: any) => nodes.find((n) => n.id === d.target)?.y || 0)
      .attr('stroke', '#999')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,5')

    // Draw nodes
    const node = g
      .selectAll('.node')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)

    node
      .append('circle')
      .attr('r', (d) => (d.type === 'cloud' ? 20 : d.type === 'fog' ? 15 : 10))
      .attr('fill', (d) =>
        d.type === 'cloud' ? '#1976d2' : d.type === 'fog' ? '#2e7d32' : '#ed6c02'
      )
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)

    node
      .append('text')
      .attr('dy', 30)
      .attr('text-anchor', 'middle')
      .text((d) => d.name)
      .style('font-size', '10px')
      .style('fill', '#333')

  }, [networkState, predictions])

  return (
    <Box display="flex" justifyContent="center" alignItems="center" height="100%">
      <svg ref={svgRef}></svg>
    </Box>
  )
}

export default NetworkTopology
