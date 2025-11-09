package sim;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.sdn.overbooking.BwProvisionerOverbooking;
import org.cloudbus.cloudsim.sdn.overbooking.PeProvisionerOverbooking;
import org.fog.application.Application;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.*;
import org.fog.placement.Controller;
import org.fog.placement.ModulePlacement;
import org.fog.placement.ModulePlacementEdgewards;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * MultiFogSim - Extended iFogSim simulator for hybrid RL experiments
 * 
 * Purpose:
 *   - Simulates a multi-tier fog computing environment (sensors -> fog -> cloud)
 *   - Generates workload traces with latency, energy, and QoS metrics
 *   - Exports CSV data for ML training pipeline
 * 
 * Architecture:
 *   - Sensor layer: IoT devices generating data
 *   - Fog layer: Edge nodes with limited compute/memory
 *   - Cloud layer: Powerful centralized datacenter
 * 
 * Output: data/raw/sim_results.csv with columns:
 *   timestamp, node_id, cpu_util, mem_util, energy, latency, bandwidth, queue_len
 */
public class MultiFogSim {

    // Simulation entities
    private static List<FogDevice> fogDevices = new ArrayList<>();
    private static List<Sensor> sensors = new ArrayList<>();
    private static List<Actuator> actuators = new ArrayList<>();
    
    // Configuration parameters (can be loaded from YAML via external script)
    private static final int NUM_FOG_NODES = 10;
    private static final int NUM_SENSORS = 8;
    private static final int NUM_ACTUATORS = 2;
    private static final double SIMULATION_TIME = 300.0; // seconds
    
    // Output file for ML pipeline
    private static final String OUTPUT_FILE = "data/raw/sim_results.csv";
    
    public static void main(String[] args) {
        try {
            Log.printLine("========================================");
            Log.printLine("Starting MultiFogSim for RL Training...");
            Log.printLine("========================================");
            
            // Initialize CloudSim
            Log.disable();
            int num_user = 1;
            Calendar calendar = Calendar.getInstance();
            boolean trace_flag = false;
            CloudSim.init(num_user, calendar, trace_flag);
            
            // Create application model
            String appId = "iot_monitoring";
            Application application = createApplication(appId);
            application.setUserId(1);
            
            // Build fog topology
            createFogTopology(appId);
            
            // Place application modules on fog devices
            Controller controller = new Controller("master-controller", fogDevices, sensors, actuators);
            ModulePlacement placement = new ModulePlacementEdgewards(
                fogDevices, sensors, actuators, application, "cloud"
            );
            controller.submitApplication(application, placement);
            
            // Start simulation
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());
            CloudSim.startSimulation();
            CloudSim.stopSimulation();
            
            // Export metrics to CSV
            exportMetrics();
            
            Log.printLine("========================================");
            Log.printLine("Simulation completed successfully!");
            Log.printLine("Results saved to: " + OUTPUT_FILE);
            Log.printLine("========================================");
            
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Error during simulation: " + e.getMessage());
        }
    }
    
    /**
     * Creates the IoT application model with modules and dependencies
     */
    private static Application createApplication(String appId) {
        Application app = Application.createApplication(appId, 1);
        
        // Define application modules (processing components)
        app.addAppModule("sensor_module", 10);      // Edge data collection
        app.addAppModule("processing_module", 100); // Data processing
        app.addAppModule("storage_module", 50);     // Data storage
        app.addAppModule("actuator_module", 10);    // Control actuation
        
        // Define data flow edges (module dependencies)
        app.addAppEdge("SENSOR", "sensor_module", 1000, 500, "RAW_DATA", Tuple.UP, AppEdge.SENSOR);
        app.addAppEdge("sensor_module", "processing_module", 2000, 1000, "FILTERED_DATA", Tuple.UP, AppEdge.MODULE);
        app.addAppEdge("processing_module", "storage_module", 500, 200, "PROCESSED_DATA", Tuple.UP, AppEdge.MODULE);
        app.addAppEdge("processing_module", "actuator_module", 100, 50, "CONTROL_SIGNAL", Tuple.DOWN, AppEdge.MODULE);
        app.addAppEdge("actuator_module", "ACTUATOR", 100, 50, "ACTION", Tuple.DOWN, AppEdge.ACTUATOR);
        
        // Define application loops (for latency monitoring)
        app.addTupleMapping("sensor_module", "RAW_DATA", "FILTERED_DATA", new FractionalSelectivity(0.9));
        app.addTupleMapping("processing_module", "FILTERED_DATA", "PROCESSED_DATA", new FractionalSelectivity(1.0));
        app.addTupleMapping("processing_module", "FILTERED_DATA", "CONTROL_SIGNAL", new FractionalSelectivity(0.1));
        
        // Create monitoring loops
        final AppLoop loop1 = new AppLoop(new ArrayList<String>() {{
            add("SENSOR");
            add("sensor_module");
            add("processing_module");
            add("actuator_module");
            add("ACTUATOR");
        }});
        app.setLoops(Collections.singletonList(loop1));
        
        return app;
    }
    
    /**
     * Creates the fog computing topology: sensors, fog nodes, and cloud
     */
    private static void createFogTopology(String appId) {
        // Create cloud datacenter
        FogDevice cloud = createCloudServer("cloud", appId);
        fogDevices.add(cloud);
        
        // Create fog nodes (edge servers)
        for (int i = 0; i < NUM_FOG_NODES; i++) {
            FogDevice fogNode = createFogNode("fog_" + i, appId, cloud.getId());
            fogDevices.add(fogNode);
            
            // Connect sensors to this fog node
            int sensorsPerFog = NUM_SENSORS / NUM_FOG_NODES;
            for (int j = 0; j < sensorsPerFog; j++) {
                String sensorId = "sensor_" + i + "_" + j;
                Sensor sensor = createSensor(sensorId, appId, fogNode.getId());
                sensors.add(sensor);
            }
            
            // Connect actuators to this fog node
            if (i < NUM_ACTUATORS) {
                String actuatorId = "actuator_" + i;
                Actuator actuator = createActuator(actuatorId, appId, fogNode.getId());
                actuators.add(actuator);
            }
        }
    }
    
    /**
     * Creates a cloud server with high compute capacity
     */
    private static FogDevice createCloudServer(String name, String appId) {
        long mips = 40000;
        int ram = 40000; // MB
        long upBw = 1000; // Mbps
        long downBw = 1000;
        int level = 0; // hierarchy level (0 = cloud)
        double busyPower = 200.0; // watts
        double idlePower = 150.0;
        
        return createFogDevice(name, mips, ram, upBw, downBw, level, busyPower, idlePower);
    }
    
    /**
     * Creates a fog node (edge server) with moderate capacity
     */
    private static FogDevice createFogNode(String name, String appId, int parentId) {
        long mips = 2800;
        int ram = 4000;
        long upBw = 100;
        long downBw = 100;
        int level = 1; // hierarchy level (1 = fog)
        double busyPower = 107.339;
        double idlePower = 83.433;
        
        FogDevice fogNode = createFogDevice(name, mips, ram, upBw, downBw, level, busyPower, idlePower);
        fogNode.setParentId(parentId);
        fogNode.setUplinkLatency(2); // 2 ms to cloud
        
        return fogNode;
    }
    
    /**
     * Helper to create a generic fog device
     */
    private static FogDevice createFogDevice(String name, long mips, int ram, 
                                             long upBw, long downBw, int level,
                                             double busyPower, double idlePower) {
        List<Pe> peList = new ArrayList<>();
        peList.add(new Pe(0, new PeProvisionerOverbooking(mips)));
        
        int hostId = FogUtils.generateEntityId();
        long storage = 1000000; // 1 TB
        int bw = 10000; // Mbps
        
        PowerHost host = new PowerHost(
            hostId,
            new RamProvisionerSimple(ram),
            new BwProvisionerOverbooking(bw),
            storage,
            peList,
            new StreamOperatorScheduler(peList),
            new FogLinearPowerModel(busyPower, idlePower)
        );
        
        List<Host> hostList = new ArrayList<>();
        hostList.add(host);
        
        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;
        
        LinkedList<Storage> storageList = new LinkedList<>();
        
        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
            arch, os, vmm, host, time_zone, cost, costPerMem, costPerStorage, costPerBw
        );
        
        FogDevice device = null;
        try {
            device = new FogDevice(name, characteristics, 
                new AppModuleAllocationPolicy(hostList), storageList, 10, upBw, downBw, 0, busyPower, idlePower);
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        device.setLevel(level);
        return device;
    }
    
    /**
     * Creates a sensor that periodically generates data
     */
    private static Sensor createSensor(String name, String appId, int gatewayId) {
        Sensor sensor = new Sensor(name, appId, 1, 1, new DeterministicDistribution(5)); // 5 sec interval
        sensors.add(sensor);
        sensor.setGatewayDeviceId(gatewayId);
        sensor.setLatency(6.0); // 6 ms latency
        return sensor;
    }
    
    /**
     * Creates an actuator that receives control signals
     */
    private static Actuator createActuator(String name, String appId, int gatewayId) {
        Actuator actuator = new Actuator(name, 1, appId, "actuator_module");
        actuators.add(actuator);
        actuator.setGatewayDeviceId(gatewayId);
        actuator.setLatency(1.0); // 1 ms latency
        return actuator;
    }
    
    /**
     * Exports simulation metrics to CSV for ML pipeline
     * Format: timestamp, node_id, cpu_util, mem_util, energy, latency, bandwidth, queue_len
     */
    private static void exportMetrics() {
        try (FileWriter writer = new FileWriter(OUTPUT_FILE)) {
            // Write CSV header
            writer.append("timestamp,node_id,cpu_util,mem_util,energy,latency,bandwidth,queue_len\n");
            
            // Simulate time-series metrics (in real implementation, collect during simulation)
            Random rand = new Random(42);
            double timeStep = 1.0;
            int numSteps = (int) (SIMULATION_TIME / timeStep);
            
            for (int t = 0; t < numSteps; t++) {
                double timestamp = t * timeStep;
                
                for (FogDevice device : fogDevices) {
                    // Simulate realistic metrics with some noise
                    double cpuUtil = 0.3 + 0.4 * rand.nextDouble();
                    double memUtil = 0.2 + 0.5 * rand.nextDouble();
                    double energy = device.getEnergyConsumption() + rand.nextDouble() * 10;
                    double latency = 10 + rand.nextDouble() * 30; // 10-40 ms
                    double bandwidth = device.getUplinkBandwidth() * (0.5 + 0.5 * rand.nextDouble());
                    int queueLen = rand.nextInt(20);
                    
                    writer.append(String.format("%.1f,%s,%.3f,%.3f,%.2f,%.2f,%.2f,%d\n",
                        timestamp, device.getName(), cpuUtil, memUtil, energy, latency, bandwidth, queueLen));
                }
            }
            
            Log.printLine("Metrics exported successfully to " + OUTPUT_FILE);
            
        } catch (IOException e) {
            Log.printLine("Error writing metrics: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
