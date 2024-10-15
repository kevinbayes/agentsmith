use std::sync::{Arc, Mutex};
use crate::agent::agent::Agent;

pub struct SoftwareProject {
    id: String,
    team: Vec<Agent>,
    orchestrator: Arc<SoftwareProjectOrchestrator>,
    state: Arc<Mutex<SoftwareProjectState>>,
}


pub struct SoftwareProjectOrchestrator {

}


pub struct SoftwareProjectState {

}

