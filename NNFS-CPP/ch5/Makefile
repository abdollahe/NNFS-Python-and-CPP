# In the name of Allah
# ---- variables ----
CXX        ?= g++
SRC_DIR    := src
INC_DIR    := include
BUILD_DIR_BASE  := build
EXECUTABLE_NAME ?= output

# Build type: 'debug' or 'prod' (default = prod)
BUILD_TYPE ?= prod
CXX_VERSION ?= c++17

# Will refine BUILD_DIR and TARGET after BUILD_TYPE is known
# (TARGET depends on BUILD_DIR)
# Base compiler flags
CXXFLAGS := -std=$(CXX_VERSION) -Wall -Wextra

# Recursively add all include subdirs: -I include -I include/foo -I include/foo/bar ...
INC_DIRS := $(shell find $(INC_DIR) -type d 2>/dev/null)
CXXFLAGS += $(addprefix -I ,$(INC_DIRS))

# Adjust flags and output dir per build type
ifeq ($(BUILD_TYPE),debug)
  CXXFLAGS += -O0 -g
  BUILD_DIR := $(BUILD_DIR_BASE)/debug
else ifeq ($(BUILD_TYPE),prod)
  CXXFLAGS += -O2
  BUILD_DIR := $(BUILD_DIR_BASE)/prod
else
  $(error Unknown BUILD_TYPE '$(BUILD_TYPE)'. Use 'debug' or 'prod')
endif

TARGET := $(BUILD_DIR)/$(EXECUTABLE_NAME)

# ---- sources, objects, dependencies ----
# Recursively collect sources
SRCS := $(shell find $(SRC_DIR) -name '*.cpp' 2>/dev/null)

# Map src/foo/bar.cpp -> build/<type>/foo/bar.o
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

# ---- targets ----
.PHONY: all
all: $(TARGET)

# Link
$(TARGET): $(OBJS)
	$(CXX) -o $@ $^

# ---- compile rules for nested dirs ----
# We generate one rule per source to avoid the %/slash ambiguity in pattern rules.
# Stem list: src/foo/bar.cpp -> foo/bar
STEMS := $(patsubst $(SRC_DIR)/%.cpp,%,${SRCS})

define make-cpp-rule
$(BUILD_DIR)/$(1).o: $(SRC_DIR)/$(1).cpp
	@mkdir -p $$(@D)
	$$(CXX) $$(CXXFLAGS) -MMD -MP -c $$< -o $$@
endef

# Emit a concrete rule for each stem
$(foreach S,$(STEMS),$(eval $(call make-cpp-rule,$(S))))

# Include auto-generated header dependency files
-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR_BASE)
