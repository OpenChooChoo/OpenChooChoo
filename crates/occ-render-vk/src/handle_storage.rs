pub(crate) struct Storage<T> {
    items: Vec<T>,
}

impl<T> Storage<T> {
    pub(crate) fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub(crate) fn push(&mut self, value: T) -> u32 {
        let index = self.items.len() as u32;
        self.items.push(value);
        index
    }

    pub(crate) fn get(&self, index: u32) -> Option<&T> {
        self.items.get(index as usize)
    }

    pub(crate) fn len(&self) -> usize {
        self.items.len()
    }
}
