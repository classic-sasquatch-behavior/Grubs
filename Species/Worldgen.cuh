#pragma once

#include"external_libs.h"




namespace Substrate{

    namespace Species {

        namespace Worldgen{

            namespace State {

                static bool tectonic = false;
                static bool water = false;
                static bool rest = false;
            }

            static void init_everything(){

            }


            static void run(){

                init_everything();

                //---------------------

                //tectonic setup

                do { //tectonic main loop

                } while (State::tectonic);

                //tectonic cleanup

                //---------------------

                //water setup

                do { //water main loop

                } while (State::water);

                //water cleanup

                //---------------------

                //rest setup

                do { //rest main loop
                
                } while (State::rest);

                //rest cleanup

                //---------------------
            }

        }
    }
}