#include "camera.h"

#include <glm/fwd.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/quaternion_trigonometric.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

glm::mat4 Camera::get_view_matrix()
{
    glm::mat4 cameraTranslation = glm::translate(glm::mat4(1.0), position);
    glm::mat4 cameraRotation = get_rotation_matrix();
    return glm::inverse(cameraTranslation * cameraRotation);
}

glm::mat4 Camera::get_rotation_matrix()
{
    glm::quat pitchRotation = glm::angleAxis(pitch, glm::vec3(1, 0, 0));
    glm::quat yawRotation = glm::angleAxis(yaw, glm::vec3(0, -1, 0));

    return glm::toMat4(yawRotation) * glm::toMat4(pitchRotation);
}

void Camera::increment_velocity_multiplier(const float multiplier)
{
    velocityMultiplier += multiplier;
}

void Camera::decrement_velocity_multiplier(const float multiplier)
{
    if (velocityMultiplier >= 0.1f)
    {
        velocityMultiplier -= multiplier;
    }
}

void Camera::process_sdl_event(SDL_Event& event)
{
    if (event.type == SDL_KEYDOWN)
    {
        if (event.key.keysym.sym == SDLK_w) { velocity.z = -1 * velocityMultiplier; }
        if (event.key.keysym.sym == SDLK_s) { velocity.z = 1 * velocityMultiplier; }
        if (event.key.keysym.sym == SDLK_a) { velocity.x = -1 * velocityMultiplier; }
        if (event.key.keysym.sym == SDLK_d) { velocity.x = 1 * velocityMultiplier; }
    }

    if (event.type == SDL_KEYUP)
    {
        if (event.key.keysym.sym == SDLK_w) { velocity.z = 0; }
        if (event.key.keysym.sym == SDLK_s) { velocity.z = 0; }
        if (event.key.keysym.sym == SDLK_a) { velocity.x = 0; }
        if (event.key.keysym.sym == SDLK_d) { velocity.x = 0; }
    }

    if (event.type == SDL_MOUSEMOTION)
    {
        yaw += (float)event.motion.xrel / 200.0f;
        pitch -= (float)event.motion.yrel / 200.0f;
    }
}

void Camera::update()
{
    glm::mat4 cameraRotation = get_rotation_matrix();
    position += glm::vec3(cameraRotation * glm::vec4(velocity * 0.5f, 0.0f));
}
