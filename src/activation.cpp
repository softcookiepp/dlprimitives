///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/definitions.hpp>

namespace dlprim {
    std::string activation_equation(StandardActivations act,std::string const &variable)
    {
        switch(act) {
        case StandardActivations::identity:
            return "(" + variable + ")";
        case StandardActivations::relu:
            return "(max(("+variable + "),(dtype)(0)))";
        case StandardActivations::tanh:
            return "(tanh((" + variable + ")))";
        case StandardActivations::sigmoid:
            return "((dtype)(1) / ((dtype)(1) + exp(-("+variable+"))))";
        case StandardActivations::relu6:
            return "(min(max((" + variable + "),(dtype)(0)),(dtype)(6)))";
        }
        throw ValidationError("Internal error invalid cativation");
    }
    std::string activation_backward_equation(StandardActivations act,std::string const &dy,std::string const &y)
    {
        switch(act) {
        case StandardActivations::identity:
            return "(" + dy + ")";
        case StandardActivations::relu:
            return "(("+y+">0)?"+dy+":0)";
        case StandardActivations::tanh:
            return "((1-("+y+")*("+y+"))*("+dy+"))";
        case StandardActivations::sigmoid:
            return "(("+y+")*(1-"+y+")*("+dy+"))";
        case StandardActivations::relu6:
            return "((0<"+y+" && "+y+"<6)?"+dy+":0)";
        }
        throw ValidationError("Internal error invalid cativation");
    }
} // dlprim


