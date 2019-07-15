#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask_wtf import FlaskForm
from wtforms import TextAreaField, BooleanField
from wtforms.validators import DataRequired

class InputTextForm(FlaskForm):
    inputText = TextAreaField(validators=[DataRequired()])
    # ignoreCase = BooleanField('ignore case', default=True)