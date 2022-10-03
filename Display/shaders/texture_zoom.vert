
#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

uniform mat4 transform;

void main() {
    gl_Position = transform * vec4(aPos, 1.0f);
    TexCoord = vec2(transform * vec4(aTexCoord, 1.0f, 1.0f));
}