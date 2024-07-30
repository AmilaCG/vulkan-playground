#include <iostream>

#include "Renderer/RenderBackend.h"

int main()
{
    try
    {
        RenderBackend renderBackend;
        renderBackend.Init();
        // Render loop goes here
        renderBackend.Shutdown();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
