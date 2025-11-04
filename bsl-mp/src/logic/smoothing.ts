export class EMA {
  private a: number;
  private y: number | null = null;
  constructor(alpha = 0.4) { this.a = alpha; }
  next(x: number) {
    this.y = this.y == null ? x : this.a * x + (1 - this.a) * this.y;
    return this.y;
  }
}
