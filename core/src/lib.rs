use rand::seq::SliceRandom;
use rand::{SeedableRng, rngs::StdRng};
use std::collections::VecDeque;

#[cfg(feature = "python")]
use pyo3::pyclass;

#[derive(Default)]
#[cfg_attr(feature = "python", pyclass)]
pub struct MineSweeperCore {
    rows: i32,
    cols: i32,
    mines: i32,
    visited: Vec<Vec<bool>>, // 雷区访问状态，true 表示已访问
    board: Vec<Vec<i32>>, // 雷区信息，-1 表示地雷，0-8 表示周围地雷数量
    initialized: bool, // 本局游戏是否已初始化
}

impl MineSweeperCore {
    /// 核心new
    pub fn new(rows: i32, cols: i32, mines: i32) -> Self {
        if mines >= rows * cols {
            panic!("Mines count exceeds total cells");
        }
        Self {
            rows: rows,
            cols: cols,
            mines: mines,
            visited: vec![vec![false; cols as usize]; rows as usize],
            board: vec![vec![0; cols as usize]; rows as usize],
            initialized: false,
        }
    }

    /// 生成地雷，`seed` 用于随机数生成器的种子，`safe_area` 是安全区域
    fn generate_mines(&mut self, seed: Option<u64>, safe_area: &[(i32, i32)]) {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut rng = rand::rng();
                StdRng::from_rng(&mut rng)
            }
        };
        // let total_cells = (self.rows * self.cols) as usize;
        // 候选雷位置
        let mut candidates: Vec<(i32, i32)> = (0..self.rows)
            .flat_map(|i| (0..self.cols).map(move |j| (i, j)))
            .filter(|pos| !safe_area.contains(pos)) // 排除指定区域
            .collect();

        candidates.shuffle(&mut rng); // 随机打乱候选位置
        for (row, col) in candidates.into_iter().take(self.mines as usize) {
            self.board[row as usize][col as usize] = -1;
        }
    }

    fn calculate_board_nums(&mut self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                if self.board[row as usize][col as usize] != -1 {
                    continue;
                }
                for i in -1..=1 {
                    for j in -1..=1 {
                        let r = row + i;
                        let c = col + j;
                        if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
                            let r = r as usize;
                            let c = c as usize;
                            if self.board[r][c] != -1 {
                                self.board[r][c] += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// 获取可以不能生成雷的安全区域，周围3x3的格子
    /// - `row` 和 `col` 是点击的格子坐标
    /// - 返回一个包含安全区域格子坐标的向量
    fn get_safe_area(&self, row: i32, col: i32) -> Vec<(i32, i32)> {
        let mut safe_area = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let r = row + dx;
                let c = col + dy;
                if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
                    safe_area.push((r, c));
                }
            }
        }
        safe_area
    }

    pub fn open_cell(&mut self, row: i32, col: i32) {
        // 检查参数行、列是否在雷区大小范围内
        if row < 0 || row >= self.rows || col < 0 || col >= self.cols {
            return;
        }

        // 第一次点击，初始化本局游戏
        if !self.initialized {
            let safe_area = self.get_safe_area(row, col);
            self.generate_mines(None, &safe_area); // 生成地雷分布
            self.calculate_board_nums(); // 生成雷周围数字
            self.initialized = true; // 标记游戏已初始化
        }

        let row = row as usize;
        let col = col as usize;
        let mut queue = VecDeque::new();
        queue.push_back((row, col));

        while let Some((i, j)) = queue.pop_front() {
            if self.visited[i][j] {
                continue;
            }
            self.visited[i][j] = true;

            if self.board[i][j] == 0 {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        let r = i as i32 + dx;
                        let c = j as i32 + dy;
                        if r >= 0 && r < self.rows && c >= 0 && c < self.cols {
                            let r = r as usize;
                            let c = c as usize;
                            if !self.visited[r][c] {
                                queue.push_back((r, c));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn restart(&mut self) {
        self.visited = vec![vec![false; self.cols as usize]; self.rows as usize];
        self.board = vec![vec![0; self.cols as usize]; self.rows as usize];
        self.initialized = false; // 重置初始化状态,本局游戏未开始
    }

    /// Returns the visible board, where:
    /// - -2: Unvisited cell
    /// - -1: Mine
    /// - 0: Empty cell
    /// - 1-8: Number of adjacent mines
    pub fn get_visible_board(&self) -> Vec<Vec<i32>> {
        let mut visible_board = vec![vec![-2; self.cols as usize]; self.rows as usize];
        for i in 0..self.rows as usize {
            for j in 0..self.cols as usize {
                if self.visited[i][j] {
                    visible_board[i][j] = self.board[i][j];
                }
            }
        }
        visible_board
    }

    /// Returns the game state:
    /// - -1: Game Over (hit a mine)
    /// - 0: Game In Progress
    /// - 1: Game Won (all non-mine cells opened)
    pub fn get_game_state(&self) -> i32 {
        let mut all_cleared = true;
        for i in 0..self.rows as usize {
            for j in 0..self.cols as usize {
                if self.board[i][j] == -1 && self.visited[i][j] {
                    return -1;
                }
                if self.board[i][j] != -1 && !self.visited[i][j] {
                    all_cleared = false;
                }
            }
        }
        if all_cleared { 1 } else { 0 }
    }
}

/******************** Python绑定模块 ********************/
#[cfg(feature = "python")]
mod python {
    use super::*;
    use pyo3::prelude::*;

    #[pymethods]
    impl MineSweeperCore {
        #[new]
        #[pyo3(signature = (rows=9, cols=9, mines=10))]
        fn new_py(rows: i32, cols: i32, mines: i32) -> PyResult<Self> {
            if mines >= rows * cols {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Mines count exceeds total cells",
                ));
            }
            Ok(Self::new(rows, cols, mines))
        }

        #[pyo3(name = "reset")]
        fn reset_py(&mut self) {
            self.restart();
        }

        #[pyo3(name = "batch_step")]
        fn batch_step_py(&mut self, actions: Vec<(i32, i32)>) -> Vec<(Vec<Vec<i32>>, f32, bool)> {
            let mut results = Vec::new();
            for (row, col) in actions {
                self.open_cell(row, col);
                let new_state = self.get_game_state();

                let reward = self.calculate_reward();
                let done = new_state != 0;

                results.push((self.get_visible_board(), reward, done));
            }
            results
        }

        #[pyo3(name = "calculate_reward")]
        fn calculate_reward(&self) -> f32 {
            match self.get_game_state() {
                -1 => -5.0, // 踩雷
                1 => 10.0,  // 胜利
                _ => {
                    // 计算新增打开的格子数量
                    let opened: usize = 
                        self.visited.iter().flatten().filter(|&&v| v).count();
                    opened as f32 * 0.1
                }
            }
        }

        #[pyo3(name = "open_cell")]
        fn open_cell_py(&mut self, row: i32, col: i32) {
            self.open_cell(row, col);
        }

        #[pyo3(name = "get_hidden_board")]
        fn get_hidden_board_py(&self) -> Vec<Vec<i32>> {
            self.board.clone()
        }

        #[pyo3(name = "get_valid_actions")]
        fn get_valid_actions_py(&self) -> Vec<(i32, i32)> {
            (0..self.rows)
                .flat_map(|i| (0..self.cols).map(move |j| (i, j)))
                .filter(|&(i, j)| !self.visited[i as usize][j as usize])
                .collect()
        }

        #[pyo3(name = "get_visible_board")]
        fn get_visible_board_py(&self) -> Vec<Vec<i32>> {
            self.get_visible_board()
        }

        #[pyo3(name = "get_game_state")]
        fn get_game_state_py(&self) -> i32 {
            self.get_game_state()
        }
    }

    #[pymodule]
    fn core(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<MineSweeperCore>()?;
        Ok(())
    }
}

/******************** WASM绑定模块 ********************/
#[cfg(feature = "wasm")]
mod wasm {
    use super::*;
    use js_sys::Array;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct MineSweeperCoreWasm(MineSweeperCore);

    #[wasm_bindgen]
    impl MineSweeperCoreWasm {
        #[wasm_bindgen(constructor)]
        pub fn new(rows: i32, cols: i32, mines: i32) -> Result<MineSweeperCoreWasm, JsValue> {
            if mines >= rows * cols {
                return Err(JsValue::from_str("Mines count exceeds total cells"));
            }
            let core = MineSweeperCore::new(rows, cols, mines);
            Ok(MineSweeperCoreWasm(core))
        }

        #[wasm_bindgen]
        pub fn generate_mines(&mut self, seed: Option<u64>) {
            self.0.generate_mines(seed);
        }

        #[wasm_bindgen(js_name = reset)]
        pub fn reset_wasm(&mut self) {
            self.0.restart();
        }

        #[wasm_bindgen(js_name = openCell)]
        pub fn open_cell_wasm(&mut self, row: i32, col: i32) {
            self.0.open_cell(row, col);
        }

        #[wasm_bindgen(js_name = getVisibleBoard)]
        pub fn get_visible_board_wasm(&self) -> Array {
            self.0
                .get_visible_board()
                .into_iter()
                .map(|row| {
                    let arr = Array::new();
                    for val in row {
                        arr.push(&JsValue::from(val));
                    }
                    arr
                })
                .collect()
        }

        #[wasm_bindgen(js_name = getGameState)]
        pub fn get_game_state_wasm(&self) -> i32 {
            self.0.get_game_state()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_game() {
        let game = MineSweeperCore::default();
        assert_eq!(game.rows, 9);
        assert_eq!(game.cols, 9);
    }

    #[test]
    fn test_open_cell() {
        let mut game = MineSweeperCore::new(9, 9, 0);
        game.open_cell(4, 4);
        let board = game.get_visible_board();
        assert_eq!(board[4][4], 0);
    }

    #[test]
    fn test_game_state() {
        let mut game = MineSweeperCore::new(9, 9, 0);
        game.open_cell(0, 0);
        assert_eq!(game.get_game_state(), 1);
    }

    #[test]
    fn test_restart_reset() {
        let mut game = MineSweeperCore::new(9, 9, 10);
        game.open_cell(4, 4);
        game.restart();
        
        // 验证重启后回到未初始化状态
        assert!(!game.initialized);
        assert_eq!(game.board[4][4], 0);
    }

    #[test]
    #[cfg(feature = "python")]
    fn test_mines_exceeding_cells() {
        let result = MineSweeperCore::new_py(9, 9, 100);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "python")]
    fn test_large_board() {
        let mut game = MineSweeperCore::new_py(30, 30, 100).unwrap();
        game.open_cell_py(15, 15);
        assert_eq!(game.get_game_state_py(), 0);
    }
}
