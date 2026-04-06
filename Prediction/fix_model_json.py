"""
Fix TF.js model.json to be compatible with TF.js 4.x browser runtime.
Keras 3 exports a different JSON schema than what TF.js expects (Keras 2 format).
This script converts it.
"""

import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'tfjs_model', 'model.json')

with open(model_path, 'r') as f:
    model_data = json.load(f)

def convert_layer_config(layer):
    """Convert Keras 3 layer config to TF.js-compatible Keras 2 format."""
    class_name = layer['class_name']
    config = dict(layer['config'])
    
    # Remove DTypePolicy objects — TF.js uses simple string dtype
    if 'dtype' in config and isinstance(config['dtype'], dict):
        config['dtype'] = config['dtype'].get('config', {}).get('name', 'float32')
    
    # Convert nested initializer objects
    for key in list(config.keys()):
        val = config[key]
        if isinstance(val, dict) and 'class_name' in val:
            # Convert Keras 3 initializer format to Keras 2
            config[key] = {
                'class_name': val['class_name'],
                'config': val.get('config', {})
            }
    
    # Handle InputLayer specially — TF.js needs batch_input_shape
    if class_name == 'InputLayer':
        batch_shape = config.pop('batch_shape', None)
        if batch_shape:
            config['batch_input_shape'] = batch_shape
        # Remove keys TF.js doesn't understand
        config.pop('ragged', None)
        config.pop('optional', None)
    
    result = {
        'class_name': class_name,
        'config': config
    }
    
    # Don't include build_config — TF.js doesn't need it
    return result

# Convert model topology
original_topology = model_data['modelTopology']
keras_config = original_topology['config']

converted_layers = []
for layer in keras_config['layers']:
    converted_layers.append(convert_layer_config(layer))

# Build TF.js-compatible model topology (Keras 2 format)
new_topology = {
    'class_name': 'Sequential',
    'config': {
        'name': keras_config.get('name', 'sequential'),
        'layers': converted_layers
    },
    'keras_version': '2.15.0',
    'backend': 'tensorflow'
}

model_data['modelTopology'] = new_topology

# Fix weight names for BatchNormalization
# TF.js expects: gamma, beta, moving_mean, moving_variance (4 weights per BN layer)
# Our export has: kernel, bias, gamma, beta (which maps wrong)
# BatchNormalization weights order: gamma, beta, moving_mean, moving_variance
weights_manifest = model_data['weightsManifest'][0]['weights']
fixed_weights = []
for w in weights_manifest:
    name = w['name']
    # Fix BN weight naming: our export labels them kernel/bias/gamma/beta
    # but they should be gamma/beta/moving_mean/moving_variance
    if 'batch_normalization' in name:
        parts = name.split('/')
        layer_name = parts[0]
        weight_name = parts[1]
        
        # Map: kernel->gamma, bias->beta, gamma->moving_mean, beta->moving_variance
        mapping = {
            'kernel': 'gamma',
            'bias': 'beta',
            'gamma': 'moving_mean',
            'beta': 'moving_variance'
        }
        if weight_name in mapping:
            new_name = f"{layer_name}/{mapping[weight_name]}"
            w = dict(w)
            w['name'] = new_name
    
    fixed_weights.append(w)

model_data['weightsManifest'][0]['weights'] = fixed_weights

# Write fixed model
with open(model_path, 'w') as f:
    json.dump(model_data, f)

print(f"Fixed model.json written to: {model_path}")
print(f"Converted {len(converted_layers)} layers to TF.js format")
print(f"Fixed {len(fixed_weights)} weight entries")
