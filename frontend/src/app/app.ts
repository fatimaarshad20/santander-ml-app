import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FEATURES, Feature } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields = FEATURES;
  readonly deployedStatic = !['localhost', '127.0.0.1'].includes(window.location.hostname);
  values: Record<string, number | null> = {};
  loading = false;
  error = '';
  result: PredictionResponse | null = null;
  private api = inject(PredictionService);

  constructor() {
    for (const field of this.fields) this.values[field.name] = 0.0;
  }

  submit() {
    this.error = ''; 
    this.result = null;
    if (this.deployedStatic) {
      this.error = 'Static deployment demo: start local Flask API for live prediction.';
      return;
    }
    const payload: Record<string, number> = {};
    for (const field of this.fields) {
      const val = this.values[field.name];
      if (val === null || val === undefined || isNaN(Number(val))) {
        this.error = `Invalid numeric value for ${field.label}`;
        return;
      }
      payload[field.name] = Number(val);
    }
    this.loading = true;
    this.api.predict(payload).subscribe({
      next: res => { this.result = res; this.loading = false; },
      error: err => {
        this.error = `API Error ${err.status || ''}: ` + (err.error?.error || 'Failed to reach local Flask API');
        this.loading = false;
      }
    });
  }
}