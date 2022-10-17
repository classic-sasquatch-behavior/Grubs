#pragma once

#include"external_libs.h"
#include"Matrix/matrix.h"



typedef unsigned int gl_name;
typedef unsigned char uchar;


const static std::string path_to_shaders = fs::current_path().string() + "/Display/shaders/";

static std::string load_shader(std::string shader_name) {
	fs::path shader_path = path_to_shaders + shader_name;
	std::fstream stream(shader_path);

	const auto size = fs::file_size(shader_path);
	std::string output(size, '\0');

	stream.read(output.data(), size);
	stream.close();

	return output;
}

namespace Substrate {

    namespace Species {

        namespace computer_fish{

            namespace Parameter{
                extern bool running;
            }

        }
    }

    namespace Window {
        
        namespace Surface { //to contain the information for OpenGL

            namespace Object {
                static gl_name texture;
                    
                static gl_name vertex_buffer_object;
                static gl_name vertex_array_object;

                static gl_name vertex_shader;
                static gl_name fragment_shader;
                static gl_name shader_program;

                static gl_name element_buffer_object;

                static glm::mat4 transformation_matrix; 
                static float zoom_factor;

                static uint square_indices[] = {  // note that we start from 0!
                    0, 1, 3,   // first triangle
                    1, 2, 3    // second triangle
                };
                
                static float texture_vertices[] = {
                    1.0f,  1.0f, 0.0f,	 1.0f, 1.0f,   // top right
                    1.0f, -1.0f, 0.0f,	 1.0f, 0.0f,   // bottom right
                    -1.0f, -1.0f, 0.0f,	 0.0f, 0.0f,   // bottom left
                    -1.0f,  1.0f, 0.0f,	 0.0f, 1.0f    // top left 
                };
            }

            namespace Viewport {

                static glm::vec2 position = { 0.0, 0.0 };

                static void move() {

                }

            }


            namespace Initialize {

                static void vertex_shader(std::string source) {
                    const char* shader_src = source.c_str();
                    Object::vertex_shader = glCreateShader(GL_VERTEX_SHADER);
                    glShaderSource(Object::vertex_shader, 1, &shader_src, NULL);
                    glCompileShader(Object::vertex_shader);

                    int success;
                    char infoLog[512];
                    glGetShaderiv(Object::vertex_shader, GL_COMPILE_STATUS, &success);

                    if (!success)
                    {
                        glGetShaderInfoLog(Object::vertex_shader, 512, NULL, infoLog);
                        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
                    }
                }
                
                static void fragment_shader(std::string source) {
                    const char* shader_src = source.c_str();
                    Object::fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
                    glShaderSource(Object::fragment_shader, 1, &shader_src, NULL);
                    glCompileShader(Object::fragment_shader);

                    int success;
                    char infoLog[512];
                    glGetShaderiv(Object::fragment_shader, GL_COMPILE_STATUS, &success);

                    if (!success)
                    {
                        glGetShaderInfoLog(Object::fragment_shader, 512, NULL, infoLog);
                        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
                    }
                }

                static void shader_program() {
                    Object::shader_program = glCreateProgram();
                    glAttachShader(Object::shader_program, Object::vertex_shader);
                    glAttachShader(Object::shader_program, Object::fragment_shader);
                    glLinkProgram(Object::shader_program);
                    glUseProgram(Object::shader_program);
                    glDeleteShader(Object::vertex_shader);
                    glDeleteShader(Object::fragment_shader);

                    int success;
                    char infoLog[512];
                    glGetProgramiv(Object::shader_program, GL_LINK_STATUS, &success);

                    if (!success)
                    {
                        glGetProgramInfoLog(Object::shader_program, 512, NULL, infoLog);
                        std::cout << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
                    }
                }

                static void texture() {
                    glGenBuffers(1, &Object::vertex_buffer_object);
                    glBindBuffer(GL_ARRAY_BUFFER, Object::vertex_buffer_object);
                    //try changing GL_STATIC_DRAW to GL_DYNAMIC_DRAW (or whatever) to get rid of flickering
                    glBufferData(GL_ARRAY_BUFFER, sizeof(Object::texture_vertices), Object::texture_vertices, GL_DYNAMIC_DRAW);
                    glGenTextures(1, &Object::texture);
                    glBindTexture(GL_TEXTURE_2D, Object::texture);

                }

                static void texture_ebo() {
                    glGenBuffers(1, &Object::element_buffer_object);
                    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, Object::element_buffer_object);
                    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(Object::square_indices), Object::square_indices, GL_DYNAMIC_DRAW);
                    
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(0 * sizeof(float)));
                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
                    glEnableVertexAttribArray(1);
                }


                static void transformation_matrix() {
                    Object::transformation_matrix = glm::mat4(1.0f);
                    Object::zoom_factor = 1.0f;
                }




            };

        }

        static void zoom(double amount) {
            Surface::Object::zoom_factor -= amount /100.0f;
            Surface::Object::zoom_factor = fmin(fmax(0.1, Surface::Object::zoom_factor), 1.0f);

            Surface::Object::transformation_matrix = glm::mat4(Surface::Object::zoom_factor);
        }


        static GLFWwindow* window;

        static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
            std::cout << std::endl << "scroll detected: " << yoffset << std::endl;
            zoom(yoffset);
            std::cout << "new zoom_factor: " << Surface::Object::zoom_factor << std::endl;
        }

        static void quit(){
            std::cout << std::endl << "quitting..." << std::endl;
            Substrate::Species::computer_fish::Parameter::running = false;
        }

        static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
            switch (key) {
                case GLFW_KEY_W: std::cout << "w" << std::endl; break;
                case GLFW_KEY_A: std::cout << "a" << std::endl; break;
                case GLFW_KEY_S: std::cout << "s" << std::endl; break;
                case GLFW_KEY_D: std::cout << "d" << std::endl; break;
                case GLFW_KEY_ESCAPE:  quit(); break;
                case GLFW_KEY_Q: quit(); break;
                
                default: std::cout << "unknown key" << std::endl; break;
            }
        }

        static void open(const uint width, const uint height, std::string title) {

            glfwInit();
            window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
            glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

            glfwSetKeyCallback(window, key_callback);
            glfwSetScrollCallback(window, scroll_callback);

            glfwMakeContextCurrent(window);
            glewInit();

            Surface::Initialize::vertex_shader(load_shader("texture_zoom.vert"));
            Surface::Initialize::fragment_shader(load_shader("texture.frag"));
            Surface::Initialize::shader_program();

            Surface::Initialize::texture();
            Surface::Initialize::texture_ebo();

            Surface::Initialize::transformation_matrix();

            //glfwSwapInterval(0);

        }


        static void render(Matrix<uchar>& input) {

            uchar* data = input.data(host);

            gl_name transformLoc = glGetUniformLocation(Surface::Object::shader_program, "transform");
            glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(Surface::Object::transformation_matrix));

            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, input.dim[0], input.dim[1], 0, GL_RGB, GL_UNSIGNED_BYTE, data);

            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            glClear(GL_COLOR_BUFFER_BIT);

            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

            glfwSwapBuffers(window);

            glfwPollEvents();
        }

        static void close() {
            glfwDestroyWindow(window);
            glfwTerminate();
        }	

    }

    
}
