
public final int gridWidth = 200;
public final int gridHeight = 200;
public ArrayList<ArrayList<SWCell>> grid;
public final float dt = 0.0001; //0.0125; // From table 2
public float dx;
public float dy;
public float domainWidth;
public float domainHeight;
public float canvasDx;
public float canvasDy;

public void setup() {

  size(1000, 1000, P3D);
  pixelDensity(displayDensity());
  
  frameRate(120);
  
  // Initialize grid
  dx = (float) 0.05; // dx and dy from table 2
  dy = (float) 0.05;
  
  domainWidth = gridWidth * dx;
  domainHeight = gridHeight * dy;

  canvasDx = width / (float) gridWidth;
  canvasDy = height / (float) gridHeight;
  
  grid = new ArrayList<>();
  
  for (int row = 0; row < gridHeight; row++) {
    ArrayList<SWCell> container = new ArrayList<>();
    for (int col = 0; col < gridWidth; col++) {
      container.add(new SWCell(col * dx, row * dy));
    }
    grid.add(container);
  }
  
  // Initial conditions: Place gaussian kernel in the center
  float muX = domainWidth / 2;
  float muY = domainHeight / 2;
  float sigX = 1;
  float sigY = 1;
  float cov = 0;

  for (int row = 0; row < gridHeight; row++) {
    for (int col = 0; col < gridWidth; col++) {
      float x = col*dx;
      float y = row*dy;

      grid.get(row).get(col).h = 1 + gaussian(x, y, muX, muY, sigX, sigY, cov);
    }
  }
  
}



public void draw() {

  background(51);
  
  
  stroke(255);
  noFill();
  
  ellipse(mouseX, mouseY, 20, 20);
  
  // Update the focus area
  for (int i = 0; i < 50; i++) {
    ArrayList<ArrayList<SWCell>> nextTimestep = new ArrayList<>();
    for (int row = 0; row < gridHeight; row++) {
      ArrayList<SWCell> container = new ArrayList<>();
      for (int col = 0; col < gridWidth; col++) {
  
        // Left and right neighbour
        SWCell leftNeighbour = grid.get(row).get(max(0, col-1));
        SWCell rightNeighbour = grid.get(row).get(min(gridWidth-1, col+1));
        SWCell topNeighbour = grid.get(max(0, row-1)).get(col);
        SWCell bottomNeighbour = grid.get(min(gridHeight-1, row+1)).get(col);
        
        // Boundary conditions
        if (row == 0) {
          // Top boundary is freeflow
          topNeighbour = new SWCell(topNeighbour);
          topNeighbour.y = -dy;
          topNeighbour.v = max(0, topNeighbour.v);
        }
        else if (row == gridHeight-1) {
          // Bottom boundary is a solid wall
          bottomNeighbour = new SWCell(bottomNeighbour);
          bottomNeighbour.y = gridHeight * dy;
          bottomNeighbour.v = min(0, bottomNeighbour.v);
        }
        if (col == 0) {
          // Left boundary is a solid wall
          leftNeighbour = new SWCell(leftNeighbour);
          leftNeighbour.x = -dx;
          leftNeighbour.u = max(0, leftNeighbour.u);
        }
        else if (col == gridWidth-1) {
          // Right boundary is a solid wall
          rightNeighbour = new SWCell(rightNeighbour);
          rightNeighbour.x = gridWidth * dx;
          rightNeighbour.u = min(0, rightNeighbour.u);
        }
        
        SWCell subject = new SWCell(grid.get(row).get(col));
  
        float u_updated = subject.updateShorewardMomentum(leftNeighbour, rightNeighbour, topNeighbour, bottomNeighbour, dt);
        float v_updated = subject.updateAlongshoreMomentum(leftNeighbour, rightNeighbour, topNeighbour, bottomNeighbour, dt);
        float h_updated = subject.updateWaterThickness(leftNeighbour, rightNeighbour, topNeighbour, bottomNeighbour, dt);
        
        subject.u = u_updated;
        subject.v = v_updated;
        subject.h = h_updated;
        
        container.add(subject);
        
      }
      nextTimestep.add(container);
    }
    grid = nextTimestep;
  }
  
  // Draw focus area
  
  noStroke();
  color sediment = color(107, 73, 49);
  color water = color(112, 170, 224);
  color red = color(240, 20, 20);
  float maxWaterThickness = 4;
  
  for (int row = 0; row < gridHeight; row++) {
    for (int col = 0; col < gridWidth; col++) {
      SWCell subject = grid.get(row).get(col);
      
      if (subject.v < 0) {
        fill(lerpColor(color(240, 240, 240), red, -subject.v));
      } else {
        fill(lerpColor(color(240, 240, 240), water, subject.v));
      }
      
      //float blueness = subject.u / maxWaterThickness;
      //fill(lerpColor(sediment, water, blueness));
      
      rect(col*canvasDx, row*canvasDy, canvasDx, canvasDy);
      
      //fill(255);
      //textSize(8);
      //text(subject.h, col*canvasDx, row*canvasDy + 12);
      //text(subject.u, col*canvasDx, row*canvasDy + 24);
      //text(subject.v, col*canvasDx, row*canvasDy + 36);
    }
  }

}



public float gaussian(float x, float y, float muX, float muY, float sigX, float sigY, float cov) {
  return exp(-(sigX*pow(x - muX, 2) + 2*cov*(x - muX)*(y - muY) + sigY*pow(y - muY, 2)));
}
