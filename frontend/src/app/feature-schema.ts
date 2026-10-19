export type Feature = {
  name: string;
  label: string;
  kind: 'number';
  required: boolean;
};

export const FEATURES: Feature[] = Array.from({ length: 200 }, (_, i) => ({
  name: `var_${i}`,
  label: `Variable ${i} (var_${i})`,
  kind: 'number',
  required: true
}));