#pragma once

#include "vk_types.h"
#include <SDL_events.h>

class Camera
{
public:
    glm::mat4 get_view_matrix();
    glm::mat4 get_rotation_matrix();
    void increment_velocity_multiplier(float multiplier);
    void decrement_velocity_multiplier(float multiplier);
    void process_sdl_event(SDL_Event& event);
    void update();

    glm::vec3 velocity{};
    glm::vec3 position{};
    float pitch{0.0f}; // Rotation around x-axis (vertical)
    float yaw{0.0f}; // Rotation around y-axis (horizontal)

private:
    float velocityMultiplier{1.0};
};
