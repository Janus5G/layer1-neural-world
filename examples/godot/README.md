# Godot 4 integration example

Run the standalone bridge using `npm run bridge`. In Godot 4, create a Node3D for an autonomous agent, add `Layer1BrainClient.gd` as its script, and assign a target Node3D in the Inspector. The script receives a sampled action over local WebSocket and moves the Node3D within the **Godot** world. It constructs a six-element bounded sensor vector and computes a local distance-improvement reward.

This example is a **source-level integration sketch**, not a validated Godot scene or finished game. Godot Editor/runtime testing has not been performed. The sample uses simple direct movement, no collisions, navigation or real robot controls. The neural policy starts untrained; `enable_learning` defaults to false to reduce CPU work. Custom projects must adapt observations, actions and rewards.
