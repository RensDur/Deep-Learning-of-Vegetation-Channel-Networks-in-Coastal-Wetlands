public class SWCell {

  // Private properties
  private float h; // Water layer thickness
  private float S; // Sedimentary bed elevation

  // Flow velocities (depth averaged)
  private float u; // shoreward (down-slope) component u
  private float v; // alongshore (cross-slope) component v

  // Water source term (spatially & temporally constant)
  // == Mean rate at which the tidal prism drains away over a tidal period.
  private final float Hin = 0; // (float) 1e-5;

  // Minimal water layer thickness
  private final float Hc = (float) 1e-3;

  // Gravity
  private final float grav = (float) 9.81;

  // Coordinates
  public float x;
  public float y;
  
  // Constructor
  public SWCell(float x, float y) {
    this.h = 1;
    this.S = 1;
    this.u = 0;
    this.v = 0;

    this.x = x;
    this.y = y;
  }
  
  public SWCell(SWCell other) {
  
    this.h = other.h;
    this.S = other.S;
    this.u = other.u;
    this.v = other.v;

    this.x = other.x;
    this.y = other.y;
  
  }
  
  public float waterSurfaceElevation() {
    return this.h + this.S;
  }

  // Continuity Equation
  public float updateWaterThickness(SWCell left, SWCell right, SWCell top, SWCell bottom, float dt) {

    float dx = right.x - left.x;
    float dy = bottom.y - top.y;

    float duh = (right.u * right.h) - (left.u * left.h);
    float dvh = (bottom.v * bottom.h) - (top.v * top.h);

    float dh_per_dt = - (duh / dx) - (dvh / dy) + this.Hin;

    float h_updated = this.h + dh_per_dt * dt;

    // Wetting drying
    return Float.max(h_updated, this.Hc);

  }

  // Momentum equations
  public float updateShorewardMomentum(SWCell left, SWCell right, SWCell top, SWCell bottom, float dt) {

    float dx = right.x - left.x;
    float dy = bottom.y - top.y;

    float dEta = right.waterSurfaceElevation() - left.waterSurfaceElevation();
    float dux = right.u - left.u;
    float duy = bottom.u - top.u;

    float du_per_dt = -grav * (dEta / dx) - this.u * (dux / dx) - this.v * (duy / dy);

    return this.u + du_per_dt * dt;

  }

  public float updateAlongshoreMomentum(SWCell left, SWCell right, SWCell top, SWCell bottom, float dt) {

    float dx = right.x - left.x;
    float dy = bottom.y - top.y;

    float dEta = bottom.waterSurfaceElevation() - top.waterSurfaceElevation();
    float dvx = right.v - left.v;
    float dvy = bottom.v - top.v;

    float dv_per_dt = -grav * (dEta / dy) - this.u * (dvx / dx) - this.v * (dvy / dy);

    return this.v + dv_per_dt * dt;

  }

}
