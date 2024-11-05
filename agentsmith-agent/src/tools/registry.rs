use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use crate::tools::tool::Tool;

#[derive(Clone)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<Tool>>,
}

pub type SafeToolRegistry = Arc<RwLock<ToolRegistry>>;

impl ToolRegistry {

    pub fn new() -> SafeToolRegistry {
        Arc::new(RwLock::new(ToolRegistry { tools: HashMap::new() }))
    }

    pub fn register(&mut self, code: String, tool: Tool) {
        self.tools.insert(code, Arc::new(tool));
    }

    pub fn get_tool(&self, code: &str) -> Option<Arc<Tool>> {
        self.tools.get(code).cloned()
    }

    pub fn list_tools(&self) -> Vec<Arc<Tool>> {
        self.tools.values().cloned().collect()
    }

    // Implement HashMap-like methods you want to expose
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    pub fn contains_tool(&self, code: &str) -> bool {
        self.tools.contains_key(code)
    }

    // Add iterator support
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Arc<Tool>)> {
        self.tools.iter()
    }
}


