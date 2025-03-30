use std::time::{SystemTime, UNIX_EPOCH};
//use std::io;
use tokio::time;
use tokio_modbus::prelude::*;
use tract_onnx::prelude::*;
use std::error::Error;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

const MODEL_PATH: &str = "model.onnx";

struct ModelInterface {
    model: RunnableModel<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
}

impl ModelInterface {
    fn new() -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(MODEL_PATH)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    fn run_inference(&self, occupancy_data: tract_ndarray::Array2<f32>,
                     vehicle_count_data: tract_ndarray::Array2<f32>) -> TractResult<i32> {

        let inputs = tvec!(
            occupancy_data.into_tensor().into(),
            vehicle_count_data.into_tensor().into()
        );

        let result = self.model.run(inputs)?;
        let output = result[0].to_array_view::<f32>()?;

        Ok(output.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx as i32)
            .unwrap_or(0))
    }
}

fn generate_phase_combinations(min_length: i32, max_sum: i32, step: i32) -> Vec<Vec<i32>> {
    let mut combinations = Vec::new();
    for i in (min_length..max_sum).step_by(step as usize) {
        for j in (min_length..max_sum).step_by(step as usize) {
            if i + j <= max_sum {
                combinations.push(vec![i, j]);
            }
        }
    }
    combinations
}

fn compute_green_times(action_id: i32) -> (i32, i32) {
    let actions = generate_phase_combinations(5, 36, 1);
    let fallback = vec![5, 5];
    let green_times = actions.get(action_id as usize).unwrap_or(&fallback);
    (green_times[0], green_times[1])
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let model_interface = ModelInterface::new()?;

    // Create Modbus server
    let plc_socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 101)), 502);
    let sumo_socket = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192,168,1,161)), 502);

    let (client_sumo, client_plc) = tokio::join!(
        tcp::connect(sumo_socket),
        tcp::connect(plc_socket)
    );

    let mut client_sumo = tcp::connect(sumo_socket).await?;
    println!("Connected to SUMO at {}", sumo_socket);
    let mut client_plc = tcp::connect(plc_socket).await?;
    println!("Connected to PLC at {}", plc_socket);


    loop {
        // Read 6 occupancy values and 6 vehicle count values from Modbus registers
        let data_received = client_sumo.read_holding_registers(0, 1).await??;
        // println!("Received from Modbus - Data Received: {:?}", data_received);
        // Convert register values to f32
        if data_received[0] == 1 {
            let _ = client_sumo.write_single_register(0, 0).await?;
            let occupancy_values = client_sumo.read_holding_registers(1, 6).await??;
            let vehicle_count_values = client_sumo.read_holding_registers(7, 6).await??;
            let occupancy_data = tract_ndarray::Array2::from_shape_vec(
                (1, 6),
                occupancy_values.iter().map(|&x| (x as f32) / 100.0).collect()
            )?;

            let vehicle_count_data = tract_ndarray::Array2::from_shape_vec(
                (1, 6),
                vehicle_count_values.iter().map(|&x| (x as f32)).collect()
            )?;

            /*println!("Received from Modbus - Occupancy: {:?}, Vehicle Count: {:?}",
                     occupancy_data, vehicle_count_data);*/

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs_f64();
            //println!("Current time: {}", now);

            let action_id = model_interface.run_inference(occupancy_data, vehicle_count_data)?;
            //println!("Action ID: {}", action_id);

            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)?
                .as_secs_f64();
            //println!("Time after inference: {}", now);

            let (green_time1, green_time4) = compute_green_times(action_id);

            // Write green times to Modbus registers at PLC
            
            let _ = client_plc.write_single_register(0, (green_time1*1000) as u16).await?;
            let _ = client_plc.write_single_register(1, (green_time4*1000) as u16).await?;

            let _ = client_plc.write_single_coil(0, true).await?;
            
        }

        time::sleep(time::Duration::from_millis(100)).await;
    }
}