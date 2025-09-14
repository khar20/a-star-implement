document.addEventListener('DOMContentLoaded', () => {
    const designGridContainer = document.getElementById('design-grid-container');
    const dlGridContainer = document.getElementById('dl-grid-container');
    const manhattanGridContainer = document.getElementById('manhattan-grid-container');
    const findPathBtn = document.getElementById('train-find-path-btn');
    const loader = document.getElementById('loader');

    const dlNodesExpanded = document.getElementById('dl-nodes-expanded');
    const dlLength = document.getElementById('dl-length');
    const dlTime = document.getElementById('dl-time');
    const manhattanNodesExpanded = document.getElementById('manhattan-nodes-expanded');
    const manhattanLength = document.getElementById('manhattan-length');
    const manhattanTime = document.getElementById('manhattan-time');

    const ROWS = 25;
    const COLS = 25;
    const START_POS = [0, 0];
    const GOAL_POS = [ROWS - 1, COLS - 1];

    let grid = Array(ROWS).fill(null).map(() => Array(COLS).fill(0));
    let isMouseDown = false;

    function createGrid(container, isDesignGrid) {
        container.innerHTML = '';
        container.style.gridTemplateColumns = `repeat(${COLS}, 20px)`;
        for (let r = 0; r < ROWS; r++) {
            for (let c = 0; c < COLS; c++) {
                const cell = document.createElement('div');
                cell.classList.add('cell');
                cell.dataset.row = r;
                cell.dataset.col = c;

                if (r === START_POS[0] && c === START_POS[1]) cell.classList.add('start');
                if (r === GOAL_POS[0] && c === GOAL_POS[1]) cell.classList.add('goal');

                if (isDesignGrid) {
                    cell.addEventListener('mousedown', (e) => {
                        isMouseDown = true;
                        toggleWall(r, c, cell);
                        e.preventDefault();
                    });
                    cell.addEventListener('mouseover', () => {
                        if (isMouseDown) toggleWall(r, c, cell);
                    });
                }
                container.appendChild(cell);
            }
        }
    }

    function toggleWall(r, c, cell) {
        if ((r === START_POS[0] && c === START_POS[1]) || (r === GOAL_POS[0] && c === GOAL_POS[1])) return;
        grid[r][c] = grid[r][c] === 0 ? 1 : 0;
        cell.classList.toggle('wall');
    }

    function drawResult(container, path) {
        const cells = container.querySelectorAll('.cell');
        cells.forEach(cell => {
            const r = parseInt(cell.dataset.row);
            const c = parseInt(cell.dataset.col);
            cell.className = 'cell';
            if (grid[r][c] === 1) cell.classList.add('wall');
            if (r === START_POS[0] && c === START_POS[1]) cell.classList.add('start');
            if (r === GOAL_POS[0] && c === GOAL_POS[1]) cell.classList.add('goal');
        });

        if (path) {
            path.forEach(([r, c]) => {
                const isStartOrGoal = (r === START_POS[0] && c === START_POS[1]) || (r === GOAL_POS[0] && c === GOAL_POS[1]);
                if (!isStartOrGoal) {
                    const cell = container.querySelector(`[data-row='${r}'][data-col='${c}']`);
                    if (cell) cell.classList.add('path');
                }
            });
        }
    }

    findPathBtn.addEventListener('click', async () => {
        loader.style.display = 'block';
        findPathBtn.disabled = true;
        
        try {
            const response = await fetch('/path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ grid, start: START_POS, goal: GOAL_POS })
            });
            const data = await response.json();

            // draw
            drawResult(dlGridContainer, data.path_dl);
            drawResult(manhattanGridContainer, data.path_manhattan);

            // stats
            dlNodesExpanded.textContent = data.nodes_expanded_dl;
            dlLength.textContent = data.length_dl;
            dlTime.textContent = `${data.time_dl} ms`;

            manhattanNodesExpanded.textContent = data.nodes_expanded_manhattan;
            manhattanLength.textContent = data.length_manhattan;
            manhattanTime.textContent = `${data.time_manhattan} ms`;

        } catch (error) {
            console.error('Error:', error);
            alert("An error occurred. Please check the console.");
        } finally {
            loader.style.display = 'none';
            findPathBtn.disabled = false;
        }
    });

    // setup
    createGrid(designGridContainer, true);
    createGrid(dlGridContainer, false);
    createGrid(manhattanGridContainer, false);
    drawResult(dlGridContainer, null);
    drawResult(manhattanGridContainer, null);

    document.body.addEventListener('mouseup', () => isMouseDown = false);
});