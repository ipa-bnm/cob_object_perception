/*
 * texture_features.h
 *
 *  Created on: 13.01.2014
 *      Author: rbormann, Daniel Hundsdoerfer
 */

#ifndef COLOR_PARAMETER_H_
#define COLOR_PARAMETER_H_

#include <cob_texture_categorization/texture_categorization.h>
#include "texture_features.h"

class color_parameter
{
public:
	color_parameter();
	void get_color_parameter(cv::Mat img, struct feature_results *color_results);
};
#endif /* COLOR_PARAMETER_H_ */
