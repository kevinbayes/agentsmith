use crate::agent::simple_agent::TextAgent;
use crate::agent::software_architect_agent::SoftwareArchitectAgent;
use crate::agent::software_engineer_agent::SoftwareEngineerAgent;
use crate::agent::software_qa_agent::SoftwareQAAgent;
use crate::agent::software_reviewer_agent::SoftwareReviewerAgent;
use std::fmt;
use std::fmt::{Debug, Formatter};

#[derive(Clone, Debug)]
pub enum Agent {
    TextAgent(TextAgent),
    SoftwareArchitectAgent(SoftwareArchitectAgent),
    SoftwareEngineerAgent(SoftwareEngineerAgent),
    SoftwareQAAgent(SoftwareQAAgent),
    SoftwareReviewerAgent(SoftwareReviewerAgent),
}


#[derive(Debug, Clone)]
pub struct AgentConfig {
}

#[derive(Debug, Clone)]
struct AgentEnvironmentConfig {
}

pub trait Tool: Debug {
    fn debug_print(&self);
}

struct ToolBoxItem(Box<dyn Tool>);

impl Debug for ToolBoxItem {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), fmt::Error> {
        self.0.fmt(f) // Call the debug_print method inside Tool
    }
}

impl Clone for ToolBoxItem {
    fn clone(&self) -> Self {
        // Here, we would need the underlying tool type to implement Clone.
        // This is a limitation with trait objects, so we will assume that the
        // type inside the Box implements clone manually.
        ToolBoxItem(self.0.clone_box()) // Using a helper method (clone_box)
    }
}

trait ToolClone {
    fn clone_box(&self) -> Box<dyn Tool>;
}

// Implement ToolClone for types that implement both Tool and Clone
impl<T> ToolClone for T
where
    T: 'static + Tool + Clone,
{
    fn clone_box(&self) -> Box<dyn Tool> {
        Box::new(self.clone())
    }
}

// Add ToolClone to the Tool trait
impl dyn Tool {
    fn clone_box(&self) -> Box<dyn Tool> {
        self.clone_box()
    }
}

#[derive(Debug, Clone)]
pub struct AgentEnvironment {
    config: AgentEnvironmentConfig,
    tools: Vec<ToolBoxItem>
}
